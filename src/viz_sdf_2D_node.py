import io
import os
import numpy as np
import torch
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR, COLPREDMPC_TMP_DIR
from collision_predictor_mpc.controller import NMPC
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.utils.pos_sampler import PosSampler
from collision_predictor_mpc.utils.math import quat2rot
import matplotlib as mpl
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Header
from sdf_nmpc_ros.msg import Latent
from sensor_msgs.msg import Image
from nav_msgs.msg import Path


class RosWrapper:
    def __init__(self, cfg, ang=0, tol=2.5e-2, nb_points=10000):
        rospy.init_node('viz_sdf')

        self.cfg = cfg
        self.tol = tol
        self.ang = ang

        self.sdf = torch.jit.load(f'{COLPREDMPC_TMP_DIR}/{self.cfg.files.sdf}')
        self.sdf.to(self.cfg.nn.vae_device)
        self.sdf.eval()

        ## sample points in frustrum
        self.fcontour_levels = 20  # nb of displayed levels in fcontour
        self.lvlset = cfg.robot.size.xy + cfg.mpc.bound_margin # level-set to display
        self.lvl_tol = 0.02  # tolerance threshold for lvlset
        self.max_df = self.sdf.max_df
        pos_sampler = PosSampler(self.cfg.sensor.dmax, self.cfg.sensor.hfov, self.cfg.sensor.vfov, 0, device=self.cfg.nn.vae_device)
        self.points = pos_sampler.grid_frustrum_slice(nb_points, ang, False)
        self.nb_points = self.points.shape[0]
        self.shape_grid = (int(self.points.shape[0]**(1/2)),int(self.points.shape[0]**(1/2)))
        self.x = self.points[:,0].cpu().reshape(self.shape_grid).T
        self.y = self.points[:,1].cpu().reshape(self.shape_grid).T

        ## plot setup
        self.dlim = cfg.sensor.dmax
        self.tick_space = 1
        self.xlim = [-self.dlim, self.dlim]

        self.fig = plt.figure(figsize=(3,3), dpi=300)
        self.ax = self.fig.subplots(nrows=1, ncols=1)
        self.ax.invert_xaxis()
        self.ax.set_xticks(list(np.arange(-self.dlim, self.dlim+0.01, self.tick_space)) + [0])
        self.ax.set_yticks(np.arange(-self.dlim, self.dlim+0.01, self.tick_space))
        self.ax.set_aspect('equal')
        self.ax.set_axisbelow(True)
        self.ax.grid(color='gray', linestyle='dashed', alpha=0.7)
        self.ax.set_xlabel(r'y axis [m]', fontfamily='serif')
        self.ax.set_ylabel(r'x axis [m]', fontfamily='serif')
        self.ax.scatter(0, 0, color='k', marker='o', s=30, zorder=5)

        ## contour data (cannot be updated dynamically)
        self.data_contour = [None, None, None]

        ## ref / traj plots data
        self.data_ref, = self.ax.plot([], [], '-', color='black', linewidth=1.5, markersize=5, zorder=3)
        self.data_traj, = self.ax.plot([], [], '-', color='green', linewidth=2, markersize=7, zorder=4)

        ## data structures
        self.points_sdf = None
        self.path_ref = None
        self.path_traj = None

        ## colorbar setup
        # plt.subplots_adjust(left=0.25)
        # self.cbar_ax = self.fig.add_axes([0, 0.25, 0.03, 0.47])
        # self.fig.colorbar(mappable=mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1,1), cmap='magma'), cax=self.cbar_ax, orientation='vertical')
        # ticks = np.linspace(-self.max_df, self.max_df, 5)
        # self.cbar_ax.yaxis.set_ticks_position('left')
        # self.cbar_ax.set_yticks(ticks, map(str, np.linspace(-2,2,5)), rotation=90)
        # # self.cbar_ax.set_yticks(list(map(float,ticks)), ticks, rotation=270)
        # self.cbar_ax.axhline(0, c='white', linewidth=2)
        # self.cbar_ax.axhline(self.lvlset, c='blue', linewidth=2)
        # self.cbar_ax.set_label(r'Neural SDF [m]')

        ## topics
        self.sub_latent = rospy.Subscriber(self.cfg.ros.topics['latent'], Latent, self.cb_latent, tcp_nodelay=True, queue_size=1)
        self.sub_ref = rospy.Subscriber(self.cfg.ros.topics.viz['ref_horizon'], Path, self.cb_path_ref, tcp_nodelay=True, queue_size=1)
        self.sub_traj = rospy.Subscriber(self.cfg.ros.topics.viz['traj_horizon'], Path, self.cb_path_traj, tcp_nodelay=True, queue_size=1)
        self.pub_img = rospy.Publisher(self.cfg.ros.topics.viz['img_df'], Image, queue_size=1)

        rospy.loginfo('node viz_sdf_2d started successfully')
        rospy.spin()

    def gen_image(self):
        if self.points_sdf is not None:
            [c.remove() for contour in self.data_contour if contour is not None for c in contour.collections]
            self.data_contour[0] = self.ax.contourf(self.y, self.x, self.points_sdf, levels=self.fcontour_levels, vmin=-self.max_df, vmax=self.max_df, cmap='magma', zorder=0)
            self.data_contour[1] = self.ax.contourf(self.y, self.x, self.points_sdf, levels=[-self.lvl_tol,self.lvl_tol], colors='white', zorder=1)
            self.data_contour[2] = self.ax.contourf(self.y, self.x, self.points_sdf, levels=[self.lvlset-self.lvl_tol,self.lvlset+self.lvl_tol], colors='blue', zorder=2)

            if self.path_ref is not None:
                self.data_ref.set_xdata(self.path_ref[1,:])
                self.data_ref.set_ydata(self.path_ref[0,:])
            if self.path_traj is not None:
                self.data_traj.set_xdata(self.path_traj[1,:])
                self.data_traj.set_ydata(self.path_traj[0,:])

            self.fig.tight_layout()
            io_buf = io.BytesIO()
            self.fig.savefig(io_buf, format='raw', transparent=True)
            io_buf.seek(0)
            img = np.frombuffer(io_buf.getvalue(), dtype=np.uint8).reshape((int(self.fig.bbox.bounds[3]), int(self.fig.bbox.bounds[2]), -1))
            io_buf.close()

            img = np.stack((img[:,:,2], img[:,:,1], img[:,:,0]), axis=2)  # bgr

            msg = Image()
            msg.header = msg.header
            msg.height = img.shape[0]
            msg.width = img.shape[1]
            msg.encoding = '8UC3'
            msg.step = msg.width * 3
            msg.data = img.tobytes()
            self.pub_img.publish(msg)

    def cb_path_ref(self, msg):
        traj = []
        W_p_B0 = np.array([msg.poses[0].pose.position.x, msg.poses[0].pose.position.y, msg.poses[0].pose.position.z])
        B0_R_W = quat2rot(np.array([msg.poses[0].pose.orientation.w, msg.poses[0].pose.orientation.x, msg.poses[0].pose.orientation.y, msg.poses[0].pose.orientation.z])).T
        for p in msg.poses:
            W_p = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
            B0_p = B0_R_W @ (W_p - W_p_B0)
            traj.append(B0_p)
        self.path_ref = np.array(traj).T

    def cb_path_traj(self, msg):
        traj = []
        W_p_B0 = np.array([msg.poses[0].pose.position.x, msg.poses[0].pose.position.y, msg.poses[0].pose.position.z])
        B0_R_W = quat2rot(np.array([msg.poses[0].pose.orientation.w, msg.poses[0].pose.orientation.x, msg.poses[0].pose.orientation.y, msg.poses[0].pose.orientation.z])).T
        for p in msg.poses:
            W_p = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z])
            B0_p = B0_R_W @ (W_p - W_p_B0)
            traj.append(B0_p)
        self.path_traj = np.array(traj).T

    def cb_latent(self, msg):
        with torch.no_grad():
            latent = torch.tensor(msg.latent.data, dtype=torch.float32, device=self.cfg.nn.vae_device).reshape(1, -1)
            self.points_sdf = self.sdf(torch.hstack([self.points, latent.repeat(self.nb_points, 1)])).flatten().cpu().reshape(self.shape_grid).T

        self.gen_image()

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'
    tolerance = float(rospy.get_param("/tol"))
    nb_points = int(rospy.get_param("/nb_points"))

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    RosWrapper(cfg, tolerance, nb_points)
