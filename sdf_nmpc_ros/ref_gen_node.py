#!/usr/bin/env python3

import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.utils.reference import Waypoint
from collision_predictor_mpc.ref_gen import RefGen
from collision_predictor_mpc.utils.math import yaw2quat, quat2rot
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Transform, Twist, Quaternion, Vector3
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from std_srvs.srv import Trigger, SetBool
from builtin_interfaces.msg import Duration


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


class RosWrapper(Node):
    def __init__(self, cfg):
        super().__init__('ref_gen')

        self.cfg = cfg
        self.ref_gen = RefGen(self.cfg)
        self.timer_period = 1.0 / 200.0  # 200 Hz
        self.x0 = None
        self.wps = []
        self.joy_lp = LowPassFilter(self.cfg.ref.joystick_lp_alpha)
        self.joy_lp.filter([0, 0, 0, 0])
        self.t_joy = 0
        self.timeout_joy = 0.5  # [s]

        self.pub_traj = self.create_publisher(MultiDOFJointTrajectory, 
                                               self.cfg.ros.topics['ref_horizon'], 1)
        self.pub_traj_viz = self.create_publisher(Path, 
                                                   self.cfg.ros.topics.viz['ref_horizon'], 1)
        self.pub_wps = self.create_publisher(Path, 
                                              self.cfg.ros.topics['ref_wps'], 1)

        self.sub_state = self.create_subscription(Odometry, 
                                                   self.cfg.ros.topics['odom'], 
                                                   self.cb_state, 1)
        self.sub_wps = self.create_subscription(Path, 
                                                 self.cfg.ros.topics['ref_wps'], 
                                                 self.cb_wps, 1)
        self.sub_joystick = self.create_subscription(Twist, 
                                                      self.cfg.ros.topics['joystick'], 
                                                      self.cb_joystick, 1)

        self.srv_get_yaw_mode = self.create_service(Trigger, 
                                                     self.cfg.ros.srv['get_yaw_mode'], 
                                                     self.srv_get_yaw_mode_callback)
        self.srv_set_yaw_mode = self.create_service(SetBool, 
                                                     self.cfg.ros.srv['set_yaw_mode'], 
                                                     self.srv_set_yaw_mode_callback)
        self.srv_hover = self.create_service(Trigger, 
                                              self.cfg.ros.srv['hover'], 
                                              self.srv_hover_callback)
        self.srv_takeoff = self.create_service(Trigger, 
                                                self.cfg.ros.srv['takeoff'], 
                                                self.srv_takeoff_callback)
        self.srv_goto = self.create_service(Trigger, 
                                             self.cfg.ros.srv['goto'], 
                                             self.srv_goto_callback)
        self.srv_stop = self.create_service(Trigger, 
                                             self.cfg.ros.srv['stop'], 
                                             self.srv_stop_callback)

        self.get_logger().info('[ref_gen] node started successfully')
        self.state = 'start'
        self.printed = False
        
        # Create timer for state machine
        self.timer = self.create_timer(self.timer_period, self.sm_manager)

    def sm_manager(self):
        ## state machine
        if self.state == 'start':
            ref_traj = []
            if self.x0 is not None:
                if self.cfg.ref.ref_mode == 'joystick':
                    self.state = 'joystick'
                    self.get_logger().info('[ref_gen] odometry received, listening to joystick inputs')
                else:
                    self.state = 'waypoint'
                    self.get_logger().info('[ref_gen] odometry received, tracking current position')
        elif self.state == 'joystick':
            ## check if joystick command is outdated
            now = self.get_clock().now().nanoseconds / 1e9
            if now > self.t_joy + self.timeout_joy:
                self.t_joy = now
                self.joy_lp.filter([0, 0, 0, 0])
                self.get_logger().warn('[ref_gen] timeout joystick command, defaulting to 0')
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
                        self.get_logger().info('[ref_gen] last waypoint reached, hovering')
                ref_traj = self.ref_gen.gen_ref_list_wps(self.wps[:1] if self.cfg.ref.stop_and_go else self.wps)
        if ref_traj:
            self.publish(ref_traj)

    def publish(self, ref_traj):
        msg = MultiDOFJointTrajectory()
        msg_viz = Path()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.cfg.ros.frames.world)
        msg_viz.header = msg.header
        for r in ref_traj:
            pos = Vector3(x=r.p[0], y=r.p[1], z=r.p[2])
            rot = Quaternion(x=r.qz[1], y=r.qz[2], z=r.qz[3], w=r.qz[0])
            msg.points.append(MultiDOFJointTrajectoryPoint(
                transforms=[Transform(translation=pos, rotation=rot)],
                velocities=[Twist(linear=Vector3(x=r.v[0], y=r.v[1], z=r.v[2]), 
                                 angular=Vector3(x=0.0, y=0.0, z=r.wz))],
                accelerations=[Twist()],
                time_from_start=Duration(sec=0, nanosec=0)
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
        self.t_joy = self.get_clock().now().nanoseconds / 1e9
        W_R_B = quat2rot(self.x0[3:])
        B_v = W_R_B @ [msg.linear.x, msg.linear.y, msg.linear.z]
        self.joy_lp.filter([*B_v, msg.angular.z])

    def cb_state(self, msg):
        pose = msg.pose.pose
        W_p_B = [pose.position.x, pose.position.y, pose.position.z]
        W_q_B = [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
        self.x0 = np.concatenate([W_p_B, W_q_B])
        self.ref_gen.x0 = self.x0

    def srv_get_yaw_mode_callback(self, request, response):
        response.success = self.ref_gen.force_yaw_current
        response.message = ''
        return response

    def srv_set_yaw_mode_callback(self, request, response):
        self.ref_gen.force_yaw_current = request.data
        response.success = self.ref_gen.force_yaw_current
        response.message = ''
        return response

    def srv_hover_callback(self, request, response):
        ref_hover = self.ref_gen.from_x0()[0]
        pose = PoseStamped()
        pose.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.cfg.ros.frames.world)
        pose.pose.position = Vector3(x=ref_hover.p[0], y=ref_hover.p[1], z=ref_hover.p[2])
        pose.pose.orientation = Quaternion(x=ref_hover.qz[1], y=ref_hover.qz[2], 
                                          z=ref_hover.qz[3], w=ref_hover.qz[0])
        msg = Path(header=pose.header, poses=[pose])
        self.pub_wps.publish(msg)
        response.success = True
        response.message = ''
        return response

    def srv_takeoff_callback(self, request, response):
        ref_hover = self.ref_gen.from_x0()[0]
        pose = PoseStamped()
        pose.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.cfg.ros.frames.world)
        pose.pose.position = Vector3(x=ref_hover.p[0], y=ref_hover.p[1], z=self.cfg.ref.zref)
        pose.pose.orientation = Quaternion(x=ref_hover.qz[1], y=ref_hover.qz[2], 
                                          z=ref_hover.qz[3], w=ref_hover.qz[0])
        msg = Path(header=pose.header, poses=[pose])
        self.pub_wps.publish(msg)
        response.success = True
        response.message = ''
        return response

    def srv_goto_callback(self, request, response):
        msg = Path()
        msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.cfg.ros.frames.world)
        for wp in self.cfg.ref.wps:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position = Vector3(x=wp[0], y=wp[1], z=wp[2])
            q = yaw2quat(wp[3])
            pose.pose.orientation = Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
            msg.poses.append(pose)
        self.pub_wps.publish(msg)
        response.success = True
        response.message = ''
        return response

    def srv_stop_callback(self, request, response):
        self.wps = []
        response.success = True
        response.message = ''
        return response


## main
def main(args=None):
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
    
    rclpy.init(args=args)
    
    # Create temporary node to get parameter
    temp_node = Node('temp_param_node')
    temp_node.declare_parameter('cfg', '')
    cfg_param = temp_node.get_parameter('cfg').value
    temp_node.destroy_node()
    
    cfg_file = f'params_{cfg_param}.yaml'
    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    
    ros_wrapper = RosWrapper(cfg)
    rclpy.spin(ros_wrapper)
    
    ros_wrapper.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
