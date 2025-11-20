#!/usr/bin/env python3
import os
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Transform, Twist, Quaternion, Vector3, Point
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_srvs.srv import Trigger, SetBool

from sdf_nmpc.utils.config import Config
from sdf_nmpc.utils.reference import Waypoint, Ref
from sdf_nmpc.utils.math import yaw2quat, quat2rot
from sdf_nmpc.ref_gen import RefGen


class LowPassFilter:
    def __init__(self, alpha: float, x0: float=None):
        self.alpha = alpha
        self.state = None is x0 if None else x0

    def filter(self, x):
        arr = np.asarray(x) if isinstance(x, list) else x
        self.state = arr if self.state is None else self.alpha * arr + (1.0 - self.alpha) * self.state
        return self.state


class RefGenNode(Node):
    def __init__(self):
        super().__init__('ref_gen')

        ## load config
        self.declare_parameter('cfg', 'default.yaml')
        cfg_file = self.get_parameter('cfg').get_parameter_value().string_value
        self.cfg = Config(cfg_file)
        self.ref_gen = RefGen(self.cfg)
    
        self.dt = 1.0 / 200.0
        self.x0 = None
        self.wps = []
        self.joy_lp = LowPassFilter(alpha=self.cfg.ros.ref.joystick_lp_alpha, x0=[0, 0, 0, 0])
        self.t_joy = 0.0
        self.timeout_joy = 0.5  # [s]
        self.state = 'start'  # initial state

        ## topics
        self.pub_traj = self.create_publisher(MultiDOFJointTrajectory, 'horizon_ref', 1)
        self.pub_traj_viz = self.create_publisher(Path, 'viz/horizon_ref', 1)
        self.pub_wps = self.create_publisher(Path, 'wps', 1)
        self.sub_state = self.create_subscription(Odometry, 'odometry', self.cb_state, 1)
        self.sub_wps = self.create_subscription(Path, 'wps', self.cb_wps, 1)
        self.sub_joystick = self.create_subscription(Twist, 'joystick', self.cb_joystick, 1)

        ## services
        self.create_service(Trigger, 'get_yaw_mode', self.srv_get_yaw_mode)
        self.create_service(SetBool, 'set_yaw_mode', self.srv_set_yaw_mode)
        self.create_service(Trigger, 'hover', self.srv_hover)
        self.create_service(Trigger, 'takeoff', self.srv_takeoff)
        self.create_service(Trigger, 'goto', self.srv_goto)
        self.create_service(Trigger, 'stop', self.srv_stop)

        ## main loop timer
        self.timer = self.create_timer(self.dt, self.sm_tick)

        self.get_logger().info('node started successfully')

    ## state machine
    def sm_tick(self):
        if self.state == 'start':
            ref_traj = []
            if self.x0 is not None:
                if self.cfg.ros.ref.ref_mode == 'joystick':
                    self.state = 'joystick'
                    self.get_logger().info('odometry received, listening to joystick inputs')
                else:
                    self.state = 'waypoint'
                    self.get_logger().info('odometry received, tracking current position')

        elif self.state == 'joystick':
            ## check if joystick command is outdated
            now = self.get_clock().now().nanoseconds * 1e-9
            if now > self.t_joy + self.timeout_joy:
                self.t_joy = now
                self.joy_lp = LowPassFilter(alpha=self.cfg.ros.ref.joystick_lp_alpha, x0=[0, 0, 0, 0])
                self.get_logger().warn('timeout joystick command, defaulting to 0')
            ref_traj = self.ref_gen.gen_ref_joystick(self.joy_lp.state)

        elif self.state == 'waypoint':
            if not self.wps:
                ref_traj = self.ref_gen.from_x0()
            else:
                # depopulate when close enough (pos only)
                if np.linalg.norm(self.x0[:3] - self.wps[0].p) < self.cfg.ros.ref.wp_tol:
                    if len(self.wps) > 1:
                        self.wps.pop(0)
                        if len(self.wps) == 1:
                            self.get_logger().info('last waypoint reached, hovering')
                ref_traj = self.ref_gen.gen_ref_list_wps(self.wps[:1] if self.cfg.ros.ref.stop_and_go else self.wps)
        else:
            ref_traj = []

        if ref_traj:
            self.publish_traj(ref_traj)

    def publish_traj(self, ref_traj: list[Ref]):
        stamp = self.get_clock().now().to_msg()
        frame = self.cfg.ros.frames.world

        traj = MultiDOFJointTrajectory()
        traj.header = Header(stamp=stamp, frame_id=frame)

        path = Path()
        path.header = traj.header

        for r in ref_traj:
            rot = Quaternion(w=float(r.q[0]), x=float(r.q[1]), y=float(r.q[2]), z=float(r.q[3]))

            traj.points.append(MultiDOFJointTrajectoryPoint(
                transforms=[Transform(
                    translation=Vector3(x=float(r.p[0]), y=float(r.p[1]), z=float(r.p[2])),
                    rotation=rot,
                )],
                velocities=[Twist(
                    linear=Vector3(x=float(r.v[0]), y=float(r.v[1]), z=float(r.v[2])),
                    angular=Vector3(x=0., y=0., z=float(r.wz)),
                )],
                accelerations=[Twist()],
                time_from_start=Duration(sec=0, nanosec=0)
            ))

            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position = Point(x=float(r.p[0]), y=float(r.p[1]), z=float(r.p[2]))
            pose.pose.orientation = rot
            path.poses.append(pose)

        self.pub_traj.publish(traj)
        self.pub_traj_viz.publish(path)

    ## callbacks
    def cb_wps(self, msg: Path):
        self.wps = []
        for pose in msg.poses:
            self.wps.append(Waypoint(
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z],
                [pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z]
            ))

    def cb_joystick(self, msg: Twist):
        self.t_joy = self.get_clock().now().nanoseconds * 1e-9
        W_R_B = quat2rot(self.x0[3:])
        B_v = W_R_B @ [msg.linear.x, msg.linear.y, msg.linear.z]
        self.joy_lp.filter([*B_v, msg.angular.z])

    def cb_state(self, msg: Odometry):
        pose = msg.pose.pose
        W_p_B = [pose.position.x, pose.position.y, pose.position.z]
        W_q_B = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        self.x0 = np.concatenate([W_p_B, W_q_B])
        self.ref_gen.x0 = self.x0

    ## services
    def srv_get_yaw_mode(self, request: Trigger.Request, response: Trigger.Response):
        response.success = bool(self.ref_gen.force_yaw_current)
        response.message = ''
        return response

    def srv_set_yaw_mode(self, request: SetBool.Request, response: SetBool.Response):
        self.ref_gen.force_yaw_current = bool(request.data)
        response.success = self.ref_gen.force_yaw_current == bool(request.data)
        response.message = ''
        return response

    def path_from_list(self, wps: list[Waypoint]):
        msg = Path()
        msg.header = Header(
            stamp=self.get_clock().now().to_msg(), 
            frame_id=self.cfg.ros.frames.world
        )

        for wp in wps:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position = Point(x=wp.p[0], y=wp.p[1], z=wp.p[2])
            pose.pose.orientation = Quaternion(w=wp.q[0], x=wp.q[1], y=wp.q[2], z=wp.q[3])
            msg.poses.append(pose)
        
        return msg

    def srv_hover(self, request: Trigger.Request, response: Trigger.Response):
        ref_hover = self.ref_gen.from_x0()[:1]
        self.pub_wps.publish(self.path_from_list(ref_hover))
        response.success = True
        response.message = ''
        return response

    def srv_takeoff(self, request: Trigger.Request, response: Trigger.Response):
        ref_to = self.ref_gen.from_x0()[:1]
        ref_to[0].p[2] = self.cfg.ref.zref  ## goto desired takeoff z
        self.pub_wps.publish(self.path_from_list(ref_to))
        response.success = True
        response.message = ''
        return response

    def srv_goto(self, request: Trigger.Request, response: Trigger.Response):
        wps = [Waypoint(wp[:3], yaw2quat(wp[3])) for wp in self.cfg.ros.ref.wps]
        
        self.pub_wps.publish(self.path_from_list(wps))
        response.success = True
        response.message = ''
        return response

    def srv_stop(self, request: Trigger.Request, response: Trigger.Response):
        self.pub_wps.publish(self.path_from_list([]))
        response.success = True
        response.message = ''
        return response


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)

    rclpy.init()
    node = RefGenNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
