import os
import numpy as np
import torch
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR, COLPREDMPC_TMP_DIR
from collision_predictor_mpc.controller import NMPC
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.utils.pos_sampler import PosSampler
from collision_predictor_mpc.utils.visualization import Imgs2Points
import rospy
from std_msgs.msg import Header
from sdf_nmpc_ros.msg import Latent
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2


class RosWrapper:
    def __init__(self, cfg, tol=2.5e-2, nb_points=10000):
        rospy.init_node('viz_sdf')

        self.cfg = cfg
        self.tol = tol
        self.lvlset = cfg.robot.size.xy + cfg.mpc.bound_margin # level-set to display

        self.sdf = torch.jit.load(f'{COLPREDMPC_TMP_DIR}/{self.cfg.files.sdf}')
        self.sdf.to(self.cfg.nn.vae_device)
        self.sdf.eval()

        img_to_points = Imgs2Points(
            False, self.cfg.sensor.is_spherical, self.cfg.sensor.dmax, self.cfg.sensor.hfov,
            self.cfg.sensor.vfov, remove_d0=True, remove_dmax=True, downsamp=2, device=self.cfg.nn.vae_device
        )

        ## sample points in frustrum
        pos_sampler = PosSampler(self.cfg.sensor.dmax * 0.95, self.cfg.sensor.hfov, self.cfg.sensor.vfov, 0, device=self.cfg.nn.vae_device)
        self.points = pos_sampler.grid_frustrum(nb_points, add_margin=False)
        self.nb_points = self.points.shape[0]

        ## topics
        topics = self.cfg.ros.topics
        self.sub_latent = rospy.Subscriber(topics['latent'], Latent, self.cb_latent, tcp_nodelay=True, queue_size=1)
        self.pub_pc = rospy.Publisher(topics['pc_df'], PointCloud2, queue_size=1)

        rospy.loginfo('node viz_sdf_3d started successfully')
        rospy.spin()

    def pc_to_msg(self, header, pc, norm=0.0, val=0.0):
        data = np.zeros([pc.shape[0], 5], dtype=np.float32)
        data[:, :3] = pc
        data[:, 3] = norm
        data[:, 4] = val
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='distance', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.sensor)
        return point_cloud2.create_cloud(header, fields, data)

    def cb_latent(self, msg):
        with torch.no_grad():
            latent = torch.tensor(msg.latent.data, dtype=torch.float32, device=self.cfg.nn.vae_device).reshape(1, -1)
            points_sdf = self.sdf(torch.hstack([self.points, latent.repeat(self.nb_points, 1)])).flatten()

            ## get lvlset
            idx = (points_sdf - self.lvlset).abs() < self.tol
            pc_lvlset = self.points[idx]
            norms = torch.norm(pc_lvlset, dim=1)

            ## publish
            msg = self.pc_to_msg(msg.header, pc_lvlset.cpu().numpy(), norm=norms.cpu().numpy())
            self.pub_pc.publish(msg)


if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'
    tolerance = float(rospy.get_param("/tol"))
    nb_points = int(rospy.get_param("/nb_points"))

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    RosWrapper(cfg, tolerance, nb_points)
