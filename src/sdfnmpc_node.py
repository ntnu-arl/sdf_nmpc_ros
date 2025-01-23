import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.controller import NMPC
from collision_predictor_mpc.ref_gen import RefGen
from collision_predictor_mpc.utils.math import euler2rot, quat2rot, rot2quat, quat2euler, euler2quat, quat2yaw
import rospy
import collections
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Float32, Float32MultiArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Quaternion, Vector3
from mavros_msgs.msg import PositionTarget, AttitudeTarget
from nav_msgs.msg import Path
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse
import time
import os


class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('nmpc')

        ## flags, message queues and class members
        self.cfg = cfg
        self.nmpc = NMPC(cfg)
        self.ref_gen = RefGen(self.cfg)
        self.rate = rospy.Rate(1/self.cfg.mpc.control_loop_time*1e3)

        self.start = False
        self.goto = False
        self.sdf_flag = False

        self.obs_queue = collections.deque(maxlen=1)
        self.state_queue = collections.deque(maxlen=1)

        self.x0 = None
        # self.ref = None
        self.ref_traj = None
        self.wps = [[0,0,1], [0,1,1], [1,1,1], [2,2,1]]

        topics = self.cfg.ros.topics
        if self.cfg.flags['simulation']:
            self.pub_cmd = rospy.Publisher(topics['cmd'], Twist, tcp_nodelay=True, queue_size=1)
        if not self.cfg.flags['simulation']:
            self.pub_cmd = rospy.Publisher(topics['cmd'], PositionTarget, tcp_nodelay=True, queue_size=1)
        self.pub_cmd_viz = rospy.Publisher(topics['cmd_viz'], TwistStamped, tcp_nodelay=False, queue_size=1)
        self.pub_cmd_traj = rospy.Publisher(topics['traj_horizon'], Path, tcp_nodelay=False, queue_size=1)
        self.pub_cmd_reftraj = rospy.Publisher('/nmpc/ref_horizon', Path, tcp_nodelay=False, queue_size=1)
        self.pub_wp_dir = rospy.Publisher(topics['wp_dir'], Marker, tcp_nodelay=False, queue_size=1)
        self.pub_cpt = rospy.Publisher(topics['cpt'], Float32, tcp_nodelay=False, queue_size=1)
        self.pub_speed = rospy.Publisher(topics['speed'], Float32, tcp_nodelay=False, queue_size=1)
        self.pub_sdf = rospy.Publisher(topics['sdf_pred'], Float32, tcp_nodelay=False, queue_size=1)

        self.sub_latent = rospy.Subscriber(topics['latent'], Float32MultiArray, self.cb_latent, tcp_nodelay=True, queue_size=1)
        self.sub_state = rospy.Subscriber(topics['odom'], Odometry, self.cb_state, tcp_nodelay=True, queue_size=1)

        rospy.Service(self.cfg.ros.srv['start'], SetBool, self.srv_startstop)
        rospy.Service(self.cfg.ros.srv['goto'], Trigger, self.srv_goto)
        rospy.Service(self.cfg.ros.srv['flag'], Trigger, self.srv_sdf)

    def spin(self):
        while not rospy.is_shutdown():
            if self.start:
                self.control_iteration()
                self.publish_stuff()
            try:
                self.rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException as e:
                # rospy.logwarn(e)
                pass

    def control_iteration(self):
        ## init and solve
        self.nmpc.set_sdf_flag(self.sdf_flag)
        self.nmpc.set_x0(self.x0)
        self.ref_gen.x0 = self.x0

        ## reference
        # p_des = [0,0,0]
        # if not self.goto:
        #     p_des[:] = self.x0[:3]
        #     if not self.cfg.flags['use_current_z']:
        #         p_des[2] = self.cfg.ref.zref
        # else:
        #     if self.cfg.ref.p_des_body:
        #         p_des = p0 + np.array(quat2rot(q0) @ np.array(self.cfg.ref.p_des).reshape(-1,1)).flatten()
        #     else:
        #         p_des = np.array(self.cfg.ref.p_des)

        if len(self.wps) > 1 and np.linalg.norm(self.x0[:3] - np.array(self.wps[0])) < 0.2:
            self.wps = self.wps[1:]
            vec = np.random.randn(3)
            # vec[2] = 0
            vec /= np.linalg.norm(vec)
            random_norm = np.random.uniform(0.2, 3)
            self.wps.append(vec * random_norm)

        self.ref_traj = self.ref_gen.gen_ref_list_wps(self.wps)
        # self.ref_traj = self.ref_gen.gen_ref_list_wps(self.wps[:1])
        # self.ref_traj = self.ref_gen.gen_ref_list_wps([p_des])
        for k, ref in enumerate(self.ref_traj):
            self.nmpc.set_ref(ref, k)

        fail_count = self.nmpc.solve()
        if self.cfg.mpc.max_solver_fail and fail_count == self.cfg.mpc.max_solver_fail:
            print('NMPC FAILED. SENDING HOVER COMMAND.')
            self.send_commands(self.nmpc.cmd_hover)
            self.start = False
            self.goto = False
            self.sdf_flag = False
            self.nmpc.reset()
            p_hover = None
            p_des = None

    def publish_stuff(self):
        ## speed
        self.pub_speed.publish(Float32(np.linalg.norm(self.x0[7:10])))
        ## nmpc solving time
        self.pub_cpt.publish(Float32(self.nmpc.ocp.get_t()))
        ## neural sdf value
        self.pub_sdf.publish(Float32(self.nmpc.eval(0)[0]))

        ## predicted ref traj
        msg = Path()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
        for p in [self.x0[:3]] + self.wps:
        # for r in self.ref_traj:
            pose = PoseStamped()
            pose.header = msg.header
            # pose.pose.position.x = r.p[0]
            # pose.pose.position.y = r.p[1]
            # pose.pose.position.z = r.p[2]
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = p[2]
            pose.pose.orientation.w = 1
            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0
            msg.poses.append(pose)
        self.pub_cmd_reftraj.publish(msg)

        ## predicted traj
        msg = Path()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
        for p, q in self.nmpc.get_openloop_traj():
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = p[2]
            pose.pose.orientation.w = q[0]
            pose.pose.orientation.x = q[1]
            pose.pose.orientation.y = q[2]
            pose.pose.orientation.z = q[3]
            msg.poses.append(pose)
        self.pub_cmd_traj.publish(msg)

        ## command
        commands = self.nmpc.get_cmd()
        if self.cfg.flags['simulation']:
            msg = Twist()
            msg.linear.x = commands[0]
            msg.linear.y = commands[1]
            msg.linear.z = commands[2]
            msg.angular.x = 0.
            msg.angular.y = 0.
            msg.angular.z = commands[3]
        else:
            msg = PositionTarget()
            msg.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.body)
            msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
            msg.type_mask = PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ \
                            + PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                            + PositionTarget.IGNORE_YAW
            msg.acceleration_or_force.x = commands[0]
            msg.acceleration_or_force.y = commands[1]
            msg.acceleration_or_force.z = commands[2]
            msg.yaw_rate = commands[3]
        self.pub_cmd.publish(msg)

        msg_viz = TwistStamped()
        msg_viz.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.body)
        msg_viz.twist.linear.x = commands[0]
        msg_viz.twist.linear.y = commands[1]
        msg_viz.twist.linear.z = commands[2]
        msg_viz.twist.angular.x = 0
        msg_viz.twist.angular.y = 0
        msg_viz.twist.angular.z = commands[3]
        self.pub_cmd_viz.publish(msg_viz)

    def cb_latent(self, msg):
        if self.x0 is not None:
            self.nmpc.set_latent(msg.data, self.x0[:3], quat2rot(self.x0[3:7]))

    def cb_state(self, msg):
        pose = msg.pose.pose
        vel = msg.twist.twist.linear
        avel = msg.twist.twist.angular

        W_p_B = [pose.position.x, pose.position.y, pose.position.z]
        W_q_B = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        B_v_B = [vel.x, vel.y, vel.z]
        B_w_B = [avel.x, avel.y, avel.z]

        self.x0 = np.concatenate([W_p_B, W_q_B, B_v_B, B_w_B])
        # self.state_queue.appendleft(np.concatenate([W_p_B, W_R_B, B_v_B, B_w_B]))

    def srv_startstop(self, msg):
        self.start = msg.data
        return SetBoolResponse(success=True, message='')

    def srv_goto(self, msg):
        self.goto = not self.goto
        return TriggerResponse(success=True, message='')

    def srv_sdf(self, msg):
        if (self.nmpc.p[:,self.cfg.mpc.p_idx.W_p_Co]).any():
            self.sdf_flag = not self.sdf_flag
            return TriggerResponse(success=True, message='')
        else:
            return TriggerResponse(success=False, message='')


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
    cfg_file = rospy.get_param('/cfg_file')

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    ros_wrapper = RosWrapper(cfg)

    ros_wrapper.spin()
    print('ROSPY DEAD. EXITING.')
    exit(0)
