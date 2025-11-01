#!/usr/bin/env python3
import os
import collections
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from builtin_interfaces.msg import Time
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Vector3, Point, Quaternion
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from mavros_msgs.msg import PositionTarget
# from mav_msgs.msg import Actuators
from std_srvs.srv import Trigger, SetBool

from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.controller import NMPC
from collision_predictor_mpc.utils.reference import Ref
from collision_predictor_mpc.utils.math import quat2rot
from sdf_nmpc_ros.msg import Latent


def time_to_sec(t: Time) -> float:
    return float(t.sec) + float(t.nanosec) * 1e-9


class SdfNmpcNode(Node):
    def __init__(self):
        super().__init__('sdf_nmpc')

        ## load config
        self.declare_parameter('cfg', 'default.yaml')
        cfg_file = self.get_parameter('cfg').get_parameter_value().string_value
        self.cfg = Config(cfg_file)
        self.nmpc = NMPC(self.cfg)

        ctrl_dt = float(self.cfg.mpc.control_loop_time) * 1e-3
        self.state_queue = collections.deque(maxlen=10)

        self.x0 = None
        self.ref = None
        self.reset()
        self.t_img = 0
        self.t_ref = 0

        ## topics
        if self.cfg.flags['simulation']:
            if self.cfg.ros.control_interface == 'acc':
                self.pub_cmd = self.create_publisher(Twist, 'cmd/acc', 1)
            elif self.cfg.ros.control_interface == 'TRPYr':
                self.pub_cmd = self.create_publisher(Quaternion, 'cmd/att', 1)
            # elif self.cfg.ros.control_interface == 'props':
            #     self.pub_cmd = self.create_publisher(Actuators, 'cmd/props', 1)
            else:
                raise AssertionError('Unknown control_interface for simulation')
        else:
            if self.cfg.ros.control_interface == 'acc':
                self.pub_cmd = self.create_publisher(PositionTarget, 'cmd/acc', 1)
            else:
                raise AssertionError('control_interface not implemented on hardware')

        self.pub_cpt  = self.create_publisher(Float32, 'output/cpt', 1)
        self.pub_speed= self.create_publisher(Float32, 'output/speed', 1)
        self.pub_sdf  = self.create_publisher(Float32, 'output/sdf_pred', 1)

        self.pub_cmd_viz  = self.create_publisher(TwistStamped, 'viz/cmd', 1)
        self.pub_cmd_traj = self.create_publisher(Path, 'viz/horizon_traj', 1)

        self.sub_latent = self.create_subscription(Latent, 'latent', self.cb_latent, 1)
        self.sub_state  = self.create_subscription(Odometry, 'odometry', self.cb_state, 1)
        self.sub_ref    = self.create_subscription(MultiDOFJointTrajectory, 'horizon_ref', self.cb_ref, 1)

        ## services
        self.create_service(Trigger, 'get_flag', self.srv_get_flag)
        self.create_service(SetBool, 'set_flag', self.srv_set_flag)

        ## state machine
        self.running = False
        self.timer = self.create_timer(ctrl_dt, self.sm_tick)
        self.get_logger().info('node started successfully')

    ## helpers
    def now_sec(self) -> float:
        n = self.get_clock().now()
        return float(n.nanoseconds) * 1e-9

    def reset(self):
        self.sdf_flag = False
        self.failed = True
        self.nmpc.reset()

    def control_iteration(self):
        self.nmpc.set_sdf_flag(self.sdf_flag)
        self.nmpc.set_x0(self.x0)
        fail_count = self.nmpc.solve()
        self.failed = (fail_count == self.cfg.mpc.max_solver_fail)

    ## state machine
    def sm_tick(self):
        if not self.running:
            if self.x0 is not None and self.ref is not None:
                self.running = True
                self.get_logger().info('first reference received, starting mpc')
        else:
            now = self.now_sec()
            if now > self.t_ref + self.cfg.ros.timeout_ref:
                self.reset()
                self.x0 = None
                self.ref = None
                self.running = False
                self.get_logger().warn('reference timeout, resuming to idle mode')
            elif self.sdf_flag and now > self.t_img + self.cfg.ros.timeout_img:
                self.sdf_flag = False
                self.nmpc.reset_latent()
                self.get_logger().warn('observation timeout, disabling constraints')

            self.control_iteration()
            if self.failed:
                self.reset()
                self.get_logger().error('NMPC FAILED, disabling constraints')
            self.publish_cmd()
            self.publish_viz()

    ## publishers
    def publish_cmd(self):
        self.nmpc.set_x0(self.x0)
        if self.cfg.flags['simulation']:
            if self.cfg.ros.control_interface == 'acc':
                cmd_acc = self.nmpc.get_cmd_acc() if not self.failed else self.nmpc.cmd_acc_hover
                msg = Twist()
                msg.linear = Vector3(x=cmd_acc[0], y=cmd_acc[1], z=cmd_acc[2])
                msg.angular = Vector3(x=0.0, y=0.0, z=cmd_acc[3])
            elif self.cfg.ros.control_interface == 'TRPYr':
                cmd_TRPYr = self.nmpc.get_cmd_TRPYr() if not self.failed else self.nmpc.cmd_TRPYr_hover
                msg = Quaternion(w=float(cmd_trpyr[0]), x=float(cmd_trpyr[1]), y=float(cmd_trpyr[2]), z=float(cmd_trpyr[3]))
            # elif self.cfg.ros.control_interface == 'props':
            #     cmd_props = self.nmpc.get_cmd_props() if not self.failed else self.nmpc.cmd_props_hover
            #     msg = Actuators()
            #     msg.header = Header(stamp=rospy.Time.now(), frame_id='')
            #     msg.angular_velocities = cmd_props
        else:
            cmd_acc = self.nmpc.get_cmd_acc() if not self.failed else self.nmpc.cmd_acc_hover
            msg = PositionTarget()
            msg.header = Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.cfg.ros.frames.body
            )
            msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
            msg.type_mask = \
                PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ \
                + PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                + PositionTarget.IGNORE_YAW
            msg.acceleration_or_force = Vector3(x=cmd_acc[0], y=cmd_acc[1], z=cmd_acc[2])
            msg.yaw_rate = cmd_acc[3]
        self.pub_cmd.publish(msg)

    def publish_viz(self):
        ## speed
        self.pub_speed.publish(Float32(data=float(np.linalg.norm(self.x0[7:10]))))
        ## nmpc solving time
        self.pub_cpt.publish(Float32(data=float(self.nmpc.ocp.get_t())))
        ## neural sdf value
        self.pub_sdf.publish(Float32(data=float(self.nmpc.eval(0)[0])))

        if not self.failed:
            ## predicted traj
            msg_traj = Path()
            msg_traj.header = Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.cfg.ros.frames.world
            )
            for p, q in self.nmpc.get_openloop_traj():
                pose = PoseStamped()
                pose.header = msg_traj.header
                pose.pose.position = Point(x=p[0], y=p[1], z=p[2])
                pose.pose.orientation = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
                msg_traj.poses.append(pose)
            self.pub_cmd_traj.publish(msg_traj)

            ## command
            cmd_acc = self.nmpc.get_cmd_acc() if not self.failed else self.nmpc.cmd_acc_hover
            msg_cmd = TwistStamped()
            msg_cmd.header = Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.cfg.ros.frames.body
            )
            msg_cmd.twist.linear.x = cmd_acc[0]
            msg_cmd.twist.linear.y = cmd_acc[1]
            msg_cmd.twist.linear.z = cmd_acc[2]
            msg_cmd.twist.angular.x = 0.0
            msg_cmd.twist.angular.y = 0.0
            msg_cmd.twist.angular.z = cmd_acc[3]
            self.pub_cmd_viz.publish(msg_cmd)

    ## callbacks
    def cb_ref(self, msg: MultiDOFJointTrajectory):
        if self.ref is None:
            self.ref = Ref(self.cfg)
        self.t_ref = self.now_sec()
        for k, pose in enumerate(msg.points):
            self.ref.p[0] = pose.transforms[0].translation.x
            self.ref.p[1] = pose.transforms[0].translation.y
            self.ref.p[2] = pose.transforms[0].translation.z
            self.ref.q[0] = pose.transforms[0].rotation.w
            self.ref.q[1] = pose.transforms[0].rotation.x
            self.ref.q[2] = pose.transforms[0].rotation.y
            self.ref.q[3] = pose.transforms[0].rotation.z
            self.ref.v[0] = pose.velocities[0].linear.x
            self.ref.v[1] = pose.velocities[0].linear.y
            self.ref.v[2] = pose.velocities[0].linear.z
            self.ref.wz = pose.velocities[0].angular.z
            self.nmpc.set_ref(self.ref, k)

    def cb_latent(self, msg: Latent):
        if self.x0 is not None:
            # look in state history to find state closest to image timestamp
            self.t_img = time_to_sec(msg.header.stamp)
            ts, xs = zip(*self.state_queue)  # tranpose
            x = xs[np.argmin(np.abs(np.array(ts) - self.t_img))]
            self.nmpc.set_latent(msg.latent.data, x[:3], quat2rot(x[3:7]))

    def cb_state(self, msg: Odometry):
        pose = msg.pose.pose
        vel = msg.twist.twist.linear
        avel = msg.twist.twist.angular

        W_p_B = [pose.position.x, pose.position.y, pose.position.z]
        W_q_B = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        B_v_B = [vel.x, vel.y, vel.z]
        W_v_B = quat2rot(W_q_B) @ B_v_B
        B_w_B = [avel.x, avel.y, avel.z]

        self.x0 = np.concatenate([W_p_B, W_q_B, W_v_B, B_w_B], dtype=float)
        self.state_queue.appendleft((time_to_sec(msg.header.stamp), self.x0.copy()))

    # services
    def srv_get_flag(self, request: Trigger.Request, response: Trigger.Response):
        response.success = bool(self.sdf_flag)
        response.message = ''
        return response

    def srv_set_flag(self, request: SetBool.Request, response: SetBool.Response):
        if request.data and not (self.nmpc.p[0,self.cfg.mpc.p_idx.W_p_Co]).any():
            self.get_logger().error('no image received, cannot activate constraints')
            response.success = False
            response.message = ''
            return response
        else:
            self.sdf_flag = bool(request.data)
            response.success = self.sdf_flag == bool(self.sdf_flag)
            response.message = ''
            return response


## main
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
    rclpy.init()
    node = SdfNmpcNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()