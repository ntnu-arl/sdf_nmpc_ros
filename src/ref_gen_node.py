import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.utils.reference import Waypoint
from collision_predictor_mpc.ref_gen import RefGen
from collision_predictor_mpc.utils.math import yaw2quat, quat2rot
import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Transform, Twist, Quaternion, Vector3
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_srvs.srv import SetBool, SetBoolResponse


class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('ref_gen')

        self.cfg = cfg
        self.ref_gen = RefGen(self.cfg)
        self.x0 = None
        self.wps = []
        self.cmd_joy = []

        self.pub_traj = rospy.Publisher(self.cfg.ros.topics['ref_horizon'], MultiDOFJointTrajectory, tcp_nodelay=True, queue_size=1)
        self.pub_traj_viz = rospy.Publisher(self.cfg.ros.topics['ref_horizon_viz'], Path, tcp_nodelay=True, queue_size=1)

        self.sub_state = rospy.Subscriber(self.cfg.ros.topics['odom'], Odometry, self.cb_state, tcp_nodelay=True, queue_size=1)
        self.sub_wps = rospy.Subscriber(self.cfg.ros.topics['ref_wps'], Path, self.cb_wps, tcp_nodelay=True, queue_size=1)
        self.sub_joystick = rospy.Subscriber(self.cfg.ros.topics['joystick'], Twist, self.cb_joystick, tcp_nodelay=True, queue_size=1)

        rospy.Service(self.cfg.ros.srv['start'], SetBool, self.srv_startstop)

        rospy.loginfo('node ref_gen started successfully')
        rospy.spin()

    def replan(self):
        if self.wps or self.cmd_joy:
            ## depopulate wp if close enough
            if len(self.wps) > 1 and np.linalg.norm(self.x0[:3] - self.wps[0].p) < self.cfg.ref.wp_tol:
                self.wps.pop(0)

            ## plan for horizon
            self.ref_gen.x0 = self.x0
            if self.cfg.ref.ref_mode == 'topic':
                ref_traj = self.ref_gen.gen_ref_list_wps(self.wps[:1] if self.cfg.ref.stop_and_go else self.wps)
            else:
                ref_traj = self.ref_gen.gen_ref_joystick(self.cmd_joy)

            ## publish traj
            msg = MultiDOFJointTrajectory()
            msg_viz = Path()
            msg.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
            msg_viz.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
            for r in ref_traj:
                pos = Vector3(*r.p)
                rot = Quaternion(*r.qz[1:], r.qz[0])
                msg.points.append(MultiDOFJointTrajectoryPoint(
                    [Transform(pos, rot)],
                    [Twist(Vector3(*r.v), Vector3(0, 0, r.wz))],
                    [Twist()],
                    rospy.Duration.from_sec(0)
                ))
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position = pos
                pose.pose.orientation = rot
                msg_viz.poses.append(pose)
            self.pub_traj.publish(msg)
            self.pub_traj_viz.publish(msg_viz)

    def cb_wps(self, msg):
        if self.cfg.ref.ref_mode == 'topic' and self.wps:  # check if start service was called
            self.wps = []
            for pose in msg.poses:
                self.wps.append(Waypoint(
                    [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
                    [pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z]
                ))

    def cb_joystick(self, msg):
        if self.cfg.ref.ref_mode == 'joystick' and self.cmd_joy:  # check if start service was called
            self.cmd_joy = [msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z]

    def cb_state(self, msg):
        pose = msg.pose.pose
        vel = msg.twist.twist.linear
        avel = msg.twist.twist.angular

        W_p_B = [pose.position.x, pose.position.y, pose.position.z]
        W_q_B = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        B_v_B = [vel.x, vel.y, vel.z]
        # W_v_B = quat2rot(W_q_B) @ B_v_B
        B_w_B = [avel.x, avel.y, avel.z]

        self.x0 = np.concatenate([W_p_B, W_q_B, B_v_B, B_w_B])
        self.replan()

    def srv_startstop(self, msg):
        if msg.data and not self.wps and self.x0 is not None:
            p_hover = self.x0[:3]
            if not self.cfg.ref.use_current_z:
                p_hover[2] = cfg.ref.zref
            self.wps = [Waypoint(p_hover)]
            self.cmd_joy = [0, 0, 0, 0]
            rospy.loginfo(f'start service received, hovering at {p_hover}')
        elif not msg.data:
            self.wps = []
            self.cmd_joy = []
            rospy.loginfo(f'stop')
        return SetBoolResponse(success=True, message='')


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    ros_wrapper = RosWrapper(cfg)
