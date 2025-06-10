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
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse


class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.state = None

    def filter(self, input):
        if type(input) == list:
            input = np.array(input)
        if self.state is None:
            self.state = input
        else:
            self.state = self.alpha * input + (1-self.alpha) * self.state

        return self.state



class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('ref_gen')

        self.cfg = cfg
        self.ref_gen = RefGen(self.cfg)
        self.rate = rospy.Rate(200)
        self.x0 = None
        self.wps = []
        self.joy_lp = LowPassFilter(self.cfg.ref.joystick_lp_alpha)
        self.joy_lp.filter([0, 0, 0, 0])
        self.t_joy = 0
        self.timeout_joy = 0.5  # [s]

        self.pub_traj = rospy.Publisher(self.cfg.ros.topics['ref_horizon'], MultiDOFJointTrajectory, tcp_nodelay=True, queue_size=1)
        self.pub_traj_viz = rospy.Publisher(self.cfg.ros.topics.viz['ref_horizon'], Path, tcp_nodelay=True, queue_size=1)
        self.pub_wps = rospy.Publisher(self.cfg.ros.topics['ref_wps'], Path, tcp_nodelay=True, queue_size=1)

        self.sub_state = rospy.Subscriber(self.cfg.ros.topics['odom'], Odometry, self.cb_state, tcp_nodelay=True, queue_size=1)
        self.sub_wps = rospy.Subscriber(self.cfg.ros.topics['ref_wps'], Path, self.cb_wps, tcp_nodelay=True, queue_size=1)
        self.sub_joystick = rospy.Subscriber(self.cfg.ros.topics['joystick'], Twist, self.cb_joystick, tcp_nodelay=True, queue_size=1)

        rospy.Service(self.cfg.ros.srv['get_yaw_mode'], Trigger, self.srv_get_yaw_mode)
        rospy.Service(self.cfg.ros.srv['set_yaw_mode'], SetBool, self.srv_set_yaw_mode)
        rospy.Service(self.cfg.ros.srv['hover'], Trigger, self.srv_hover)
        rospy.Service(self.cfg.ros.srv['takeoff'], Trigger, self.srv_takeoff)
        rospy.Service(self.cfg.ros.srv['goto'], Trigger, self.srv_goto)
        rospy.Service(self.cfg.ros.srv['stop'], Trigger, self.srv_stop)

        rospy.loginfo('[ref_gen] node started successfully')
        self.state = 'start'
        self.sm_manager()

    def sm_manager(self):
        while not rospy.is_shutdown():
            ## sleep
            try:
                self.rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException as e:
                pass

            ## state machine
            if self.state == 'start':
                ref_traj = []
                if self.x0 is not None:
                    if self.cfg.ref.ref_mode == 'joystick':
                        self.state = 'joystick'
                        rospy.loginfo('[ref_gen] odometry received, listening to joystick inputs')
                    else:
                        self.state = 'waypoint'
                        rospy.loginfo('[ref_gen] odometry received, tracking current position')
            elif self.state == 'joystick':
                ## check if joystick command is outdated
                now = rospy.Time.now().to_sec()
                if now > self.t_joy + self.timeout_joy:
                    self.t_joy = now
                    self.joy_lp.filter([0, 0, 0, 0])
                    rospy.logwarn('[ref_gen] timeout joystick command, defaulting to 0')
                ref_traj = self.ref_gen.gen_ref_joystick(self.joy_lp.state)
            elif self.state == 'waypoint':
                if not self.wps:
                    ref_traj = self.ref_gen.from_x0()
                else:
                    ## depopulate wps if close enough (only position is considered)
                    if np.linalg.norm(self.x0[:3] - self.wps[0].p) < self.cfg.ref.wp_tol:
                        if len(self.wps) > 1:
                            self.wps.pop(0)
                        elif not self.printed:
                            self.printed = True
                            rospy.loginfo('[ref_gen] last waypoint reached, hovering')
                    ref_traj = self.ref_gen.gen_ref_list_wps(self.wps[:1] if self.cfg.ref.stop_and_go else self.wps)
            if ref_traj:
                self.publish(ref_traj)

    def publish(self, ref_traj):
        msg = MultiDOFJointTrajectory()
        msg_viz = Path()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
        msg_viz.header = msg.header
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
            pose.header = msg_viz.header
            pose.pose.position = pos
            pose.pose.orientation = rot
            msg_viz.poses.append(pose)
        self.pub_traj.publish(msg)
        self.pub_traj_viz.publish(msg_viz)

    ## callbacks and services
    def cb_wps(self, msg):
        self.printed = False
        self.wps = []
        for pose in msg.poses:
            self.wps.append(Waypoint(
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
                [pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z]
            ))

    def cb_joystick(self, msg):
        self.t_joy = rospy.Time.now().to_sec()
        W_R_B = quat2rot(self.x0[3:])
        B_v = W_R_B @ [msg.linear.x, msg.linear.y, msg.linear.z]
        self.joy_lp.filter([*B_v, msg.angular.z])

    def cb_state(self, msg):
        pose = msg.pose.pose
        W_p_B = [pose.position.x, pose.position.y, pose.position.z]
        W_q_B = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        self.x0 = np.concatenate([W_p_B, W_q_B])
        self.ref_gen.x0 = self.x0

    def srv_get_yaw_mode(self, srv):
        return TriggerResponse(success=self.ref_gen.force_yaw_current, message='')

    def srv_set_yaw_mode(self, srv):
        self.ref_gen.force_yaw_current = srv.data
        return SetBoolResponse(success=self.ref_gen.force_yaw_current, message='')

    def srv_hover(self, srv):
        ref_hover = self.ref_gen.from_x0()[0]
        pose = PoseStamped()
        pose.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
        pose.pose.position = Vector3(*ref_hover.p)
        pose.pose.orientation = Quaternion(*ref_hover.qz[1:], ref_hover.qz[0])
        msg = Path(pose.header, [pose])
        self.pub_wps.publish(msg)
        return TriggerResponse(success=True, message='')

    def srv_takeoff(self, srv):
        ref_hover = self.ref_gen.from_x0()[0]
        pose = PoseStamped()
        pose.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
        pose.pose.position = Vector3(ref_hover.p[0], ref_hover.p[1], self.cfg.ref.zref)
        pose.pose.orientation = Quaternion(*ref_hover.qz[1:], ref_hover.qz[0])
        msg = Path(pose.header, [pose])
        self.pub_wps.publish(msg)
        return TriggerResponse(success=True, message='')

    def srv_goto(self, srv):
        msg = Path()
        msg.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
        for wp in self.cfg.ref.wps:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position = Vector3(*wp[:3])
            q = yaw2quat(wp[3])
            pose.pose.orientation = Quaternion(*q[1:], q[0])
            msg.poses.append(pose)
        self.pub_wps.publish(msg)
        return TriggerResponse(success=True, message='')

    def srv_stop(self, srv):
        self.wps = []
        return TriggerResponse(success=True, message='')


## main
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    ros_wrapper = RosWrapper(cfg)
