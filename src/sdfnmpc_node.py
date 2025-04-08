import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.controller import NMPC
from collision_predictor_mpc.utils.reference import Ref
from collision_predictor_mpc.utils.math import quat2rot, euler2rot, quat2yaw
from collision_predictor_mpc.ocp import build_solver
import rospy
import collections
from std_msgs.msg import Header, Float32
from sdf_nmpc_ros.msg import Latent
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Vector3, Quaternion
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from mavros_msgs.msg import PositionTarget
from std_srvs.srv import Trigger, TriggerResponse, SetBool, SetBoolResponse


class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('sdf_nmpc')
        rospy.loginfo('starting sdf_nmpc node...')

        ## flags, message queues and class members
        self.cfg = cfg
        self.nmpc = NMPC(cfg)
        self.rate_ctrl = rospy.Rate(1/self.cfg.mpc.control_loop_time*1e3)
        self.state_queue = collections.deque(maxlen=10)

        self.x0 = None
        self.ref = None
        self.reset()
        self.t_img = 0
        self.t_ref = 0

        ## topics and services
        topics = self.cfg.ros.topics
        if self.cfg.flags['simulation']:
            if self.cfg.control_interface == 'Vacc':
                self.pub_cmd = rospy.Publisher(topics['cmd_Vacc'], Twist, tcp_nodelay=True, queue_size=1)
            else:
                self.pub_cmd = rospy.Publisher(topics['cmd_TRPYr'], Quaternion, tcp_nodelay=True, queue_size=1)
        else:
            if self.cfg.control_interface == 'Vacc':
                self.pub_cmd = rospy.Publisher(topics['cmd_Vacc'], PositionTarget, tcp_nodelay=True, queue_size=1)
            else:
                raise AssertionError('TRPYr control not interfaced with PX4')

        self.pub_cpt = rospy.Publisher(topics.output['cpt'], Float32, tcp_nodelay=True, queue_size=1)
        self.pub_speed = rospy.Publisher(topics.output['speed'], Float32, tcp_nodelay=True, queue_size=1)
        self.pub_sdf = rospy.Publisher(topics.output['sdf_pred'], Float32, tcp_nodelay=True, queue_size=1)

        self.pub_cmd_viz = rospy.Publisher(topics.viz['cmd'], TwistStamped, tcp_nodelay=True, queue_size=1)
        self.pub_cmd_traj = rospy.Publisher(topics.viz['traj_horizon'], Path, tcp_nodelay=True, queue_size=1)

        self.sub_latent = rospy.Subscriber(topics['latent'], Latent, self.cb_latent, tcp_nodelay=True, queue_size=1)
        self.sub_state = rospy.Subscriber(topics['odom_drifted' if self.cfg.flags['drifted'] else 'odom'], Odometry, self.cb_state, tcp_nodelay=True, queue_size=1)
        self.sub_ref = rospy.Subscriber(topics['ref_horizon'], MultiDOFJointTrajectory, self.cb_ref, tcp_nodelay=True, queue_size=1)

        rospy.Service(self.cfg.ros.srv['get_flag'], Trigger, self.srv_get_flag)
        rospy.Service(self.cfg.ros.srv['set_flag'], SetBool, self.srv_set_flag)

        ## start state machine
        rospy.loginfo('node sdf_nmpc started successfully')
        self.running = False
        self.sm_manager()

    def reset(self):
        self.sdf_flag = False
        self.failed = True
        self.nmpc.reset()

    def sm_manager(self):
        while not rospy.is_shutdown():
            ## sleep
            try:
                self.rate_ctrl.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException as e:
                pass

            ## state machine
            if not self.running:
                if self.x0 is not None and self.ref is not None:
                    self.running = True
                    rospy.loginfo('first reference received, starting mpc')
            else:
                now = rospy.Time.now().to_sec()
                if now > self.t_ref + self.cfg.mpc.timeout_ref:
                    self.reset()
                    self.x0 = None
                    self.ref = None
                    self.running = False
                    rospy.logwarn('reference timeout, resuming to idle mode')
                elif self.sdf_flag and now > self.t_img + self.cfg.mpc.timeout_img:
                    self.sdf_flag = False
                    self.nmpc.reset_latent()
                    rospy.logwarn('observation timeout, disabling constraints')
                self.control_iteration()
                if self.failed:
                    self.reset()
                    rospy.logerr('NMPC FAILED, disabling constraints')
                self.publish_viz()
                self.publish_cmd()

    def control_iteration(self):
        ## init and solve
        self.nmpc.set_sdf_flag(self.sdf_flag)
        self.nmpc.set_x0(self.x0)

        fail_count = self.nmpc.solve()
        if fail_count == self.cfg.mpc.max_solver_fail:
            self.failed = True
        else:
            self.failed = False

    def publish_cmd(self):
        if self.cfg.flags['simulation']:
            if self.cfg.control_interface == 'Vacc':
                cmd_Vacc = self.nmpc.get_cmd_Vacc() if not self.failed else self.nmpc.cmd_Vacc_hover
                msg = Twist()
                msg.linear = Vector3(*cmd_Vacc[:3])
                msg.angular = Vector3(0, 0, cmd_Vacc[3])
            else:
                cmd_TRPYr = self.nmpc.get_cmd_TRPYr() if not self.failed else self.nmpc.cmd_TRPYr_hover
                msg = Quaternion(*cmd_TRPYr[1:], cmd_TRPYr[0])
        else:
            cmd_Vacc = self.nmpc.get_cmd_Vacc() if not self.failed else self.nmpc.cmd_Vacc_hover
            msg = PositionTarget()
            msg.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.body)
            msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
            msg.type_mask = PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ \
                            + PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                            + PositionTarget.IGNORE_YAW
            msg.acceleration_or_force = Vector3(*cmd_Vacc[:3])
            msg.yaw_rate = cmd_Vacc[3]
        self.pub_cmd.publish(msg)

    def publish_viz(self):
        ## speed
        self.pub_speed.publish(Float32(np.linalg.norm(self.x0[7:10])))
        ## nmpc solving time
        self.pub_cpt.publish(Float32(self.nmpc.ocp.get_t()))
        ## neural sdf value
        self.pub_sdf.publish(Float32(self.nmpc.eval(0)[0]))

        if not self.failed:
            ## predicted traj
            msg = Path()
            msg.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.world)
            traj = self.nmpc.get_openloop_traj()
            for p, q in traj:
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position = Vector3(*p)
                pose.pose.orientation = Quaternion(*q[1:], q[0])
                msg.poses.append(pose)
            self.pub_cmd_traj.publish(msg)

        ## command
        cmd_Vacc = self.nmpc.get_cmd_Vacc() if not self.failed else self.nmpc.cmd_Vacc_hover
        msg_viz = TwistStamped()
        msg_viz.header = Header(stamp=rospy.Time.now(), frame_id=self.cfg.ros.frames.body)
        msg_viz.twist.linear.x = cmd_Vacc[0]
        msg_viz.twist.linear.y = cmd_Vacc[1]
        msg_viz.twist.linear.z = cmd_Vacc[2]
        msg_viz.twist.angular.x = 0
        msg_viz.twist.angular.y = 0
        msg_viz.twist.angular.z = cmd_Vacc[3]
        self.pub_cmd_viz.publish(msg_viz)

    ## callbacks and services
    def cb_ref(self, msg):
        if self.ref is None:
            self.ref = Ref(self.cfg)
        self.t_ref = rospy.Time.now().to_sec()
        for k, pose in enumerate(msg.points):
            self.ref.p[0] = pose.transforms[0].translation.x
            self.ref.p[1] = pose.transforms[0].translation.y
            self.ref.p[2] = pose.transforms[0].translation.z
            self.ref.qz[0] = pose.transforms[0].rotation.w
            self.ref.qz[1] = pose.transforms[0].rotation.x
            self.ref.qz[2] = pose.transforms[0].rotation.y
            self.ref.qz[3] = pose.transforms[0].rotation.z
            self.ref.v[0] = pose.velocities[0].linear.x
            self.ref.v[1] = pose.velocities[0].linear.y
            self.ref.v[2] = pose.velocities[0].linear.z
            self.ref.wz = pose.velocities[0].angular.z
            self.nmpc.set_ref(self.ref, k)

    def cb_latent(self, msg):
        if self.x0 is not None:
            self.t_img = msg.header.stamp.to_sec()
            ts, xs = zip(*self.state_queue)  # tranpose
            x = xs[np.argmin(np.abs(np.array(ts) - self.t_img))]
            self.nmpc.set_latent(msg.latent.data, x[:3], quat2rot(x[3:7]))

    def cb_state(self, msg):
        pose = msg.pose.pose
        vel = msg.twist.twist.linear
        avel = msg.twist.twist.angular

        W_p_B = [pose.position.x, pose.position.y, pose.position.z]
        W_q_B = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        B_v_B = [vel.x, vel.y, vel.z]
        W_v_B = quat2rot(W_q_B) @ B_v_B
        B_w_B = [avel.x, avel.y, avel.z]

        self.x0 = np.concatenate([W_p_B, W_q_B, W_v_B, B_w_B])
        self.state_queue.appendleft((msg.header.stamp.to_sec(), self.x0.copy()))

    def srv_get_flag(self, srv):
        return TriggerResponse(success=self.sdf_flag, message='')

    def srv_set_flag(self, srv):
        if srv.data and not (self.nmpc.p[0,self.cfg.mpc.p_idx.W_p_Co]).any():
            rospy.logerr('no image received, cannot activate constraints')
        else:
            self.sdf_flag = srv.data
        return SetBoolResponse(success=self.sdf_flag, message='')


## main
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

    if rospy.get_param('/rebuild'):
        path = os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file)

        rospy.loginfo(f'building solver for {path}')
        build_solver(path)
        rospy.loginfo(f'solver built')

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    ros_wrapper = RosWrapper(cfg)
    print('ROSPY DEAD. EXITING.')
    exit(0)
