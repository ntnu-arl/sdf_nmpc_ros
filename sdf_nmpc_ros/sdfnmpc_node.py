#!/usr/bin/env python3

import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.controller import NMPC
from collision_predictor_mpc.utils.reference import Ref
from collision_predictor_mpc.utils.math import quat2rot, euler2rot, quat2yaw
from collision_predictor_mpc.ocp import build_solver
import rclpy
from rclpy.node import Node
import collections
from std_msgs.msg import Header, Float32
from mav_msgs.msg import Actuators
from sdf_nmpc_ros.msg import Latent
from geometry_msgs.msg import Twist, TwistStamped, PoseStamped, Vector3, Quaternion
from nav_msgs.msg import Path, Odometry
from trajectory_msgs.msg import MultiDOFJointTrajectory
from mavros_msgs.msg import PositionTarget
from std_srvs.srv import Trigger, SetBool


class RosWrapper(Node):
    def __init__(self, cfg):
        super().__init__('sdf_nmpc')
        self.get_logger().info('[sdf_nmpc] starting node...')

        ## flags, message queues and class members
        self.cfg = cfg
        self.nmpc = NMPC(cfg)
        self.timer_period = self.cfg.mpc.control_loop_time / 1e3
        self.state_queue = collections.deque(maxlen=10)

        self.x0 = None
        self.ref = None
        self.reset()
        self.t_img = 0
        self.t_ref = 0

        ## topics and services
        topics = self.cfg.ros.topics
        if self.cfg.flags['simulation']:
            if self.cfg.control_interface == 'acc':
                self.pub_cmd = self.create_publisher(Twist, topics['cmd_acc'], 1)
            elif self.cfg.control_interface == 'TRPYr':
                self.pub_cmd = self.create_publisher(Quaternion, topics['cmd_TRPYr'], 1)
            elif self.cfg.control_interface == 'props':
                self.pub_cmd = self.create_publisher(Actuators, topics['cmd_props'], 1)
        else:
            if self.cfg.control_interface == 'acc':
                self.pub_cmd = self.create_publisher(PositionTarget, topics['cmd_acc'], 1)
            else:
                raise AssertionError('control interface not implemented on hardware')

        self.pub_cpt = self.create_publisher(Float32, topics.output['cpt'], 1)
        self.pub_speed = self.create_publisher(Float32, topics.output['speed'], 1)
        self.pub_sdf = self.create_publisher(Float32, topics.output['sdf_pred'], 1)

        self.pub_cmd_viz = self.create_publisher(TwistStamped, topics.viz['cmd'], 1)
        self.pub_cmd_traj = self.create_publisher(Path, topics.viz['traj_horizon'], 1)

        self.sub_latent = self.create_subscription(Latent, topics['latent'], self.cb_latent, 1)
        self.sub_state = self.create_subscription(Odometry, topics['odom'], self.cb_state, 1)
        self.sub_ref = self.create_subscription(MultiDOFJointTrajectory, topics['ref_horizon'], 
                                                 self.cb_ref, 1)

        self.srv_get_flag = self.create_service(Trigger, self.cfg.ros.srv['get_flag'], 
                                                 self.srv_get_flag_callback)
        self.srv_set_flag = self.create_service(SetBool, self.cfg.ros.srv['set_flag'], 
                                                 self.srv_set_flag_callback)

        ## start state machine
        self.get_logger().info('[sdf_nmpc] node started successfully')
        self.running = False
        
        # Create timer for state machine
        self.timer = self.create_timer(self.timer_period, self.sm_manager)

    def reset(self):
        self.sdf_flag = False
        self.failed = True
        self.nmpc.reset()

    def sm_manager(self):
        ## state machine
        if not self.running:
            if self.x0 is not None and self.ref is not None:
                self.running = True
                self.get_logger().info('[sdf_nmpc] first reference received, starting mpc')
        else:
            now = self.get_clock().now().nanoseconds / 1e9
            if now > self.t_ref + self.cfg.mpc.timeout_ref:
                self.reset()
                self.x0 = None
                self.ref = None
                self.running = False
                self.get_logger().warn('[sdf_nmpc] reference timeout, resuming to idle mode')
            elif self.sdf_flag and now > self.t_img + self.cfg.mpc.timeout_img:
                self.sdf_flag = False
                self.nmpc.reset_latent()
                self.get_logger().warn('[sdf_nmpc] observation timeout, disabling constraints')
            self.control_iteration()
            if self.failed:
                self.reset()
                self.get_logger().error('[sdf_nmpc] NMPC FAILED, disabling constraints')
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
        self.nmpc.set_x0(self.x0)
        if self.cfg.flags['simulation']:
            if self.cfg.control_interface == 'acc':
                cmd_acc = self.nmpc.get_cmd_acc() if not self.failed else self.nmpc.cmd_acc_hover
                msg = Twist()
                msg.linear = Vector3(x=cmd_acc[0], y=cmd_acc[1], z=cmd_acc[2])
                msg.angular = Vector3(x=0.0, y=0.0, z=cmd_acc[3])
                self.pub_cmd.publish(msg)
            elif self.cfg.control_interface == 'TRPYr':
                cmd_TRPYr = self.nmpc.get_cmd_TRPYr() if not self.failed else self.nmpc.cmd_TRPYr_hover
                msg = Quaternion(x=cmd_TRPYr[1], y=cmd_TRPYr[2], z=cmd_TRPYr[3], w=cmd_TRPYr[0])
                self.pub_cmd.publish(msg)
            elif self.cfg.control_interface == 'props':
                cmd_props = self.nmpc.get_cmd_props() if not self.failed else self.nmpc.cmd_props_hover
                msg = Actuators()
                msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id='')
                msg.angular_velocities = cmd_props
                self.pub_cmd.publish(msg)
        else:
            cmd_acc = self.nmpc.get_cmd_acc() if not self.failed else self.nmpc.cmd_acc_hover
            msg = PositionTarget()
            msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.cfg.ros.frames.body)
            msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
            msg.type_mask = PositionTarget.IGNORE_PX + PositionTarget.IGNORE_PY + PositionTarget.IGNORE_PZ \
                            + PositionTarget.IGNORE_VX + PositionTarget.IGNORE_VY + PositionTarget.IGNORE_VZ \
                            + PositionTarget.IGNORE_YAW
            msg.acceleration_or_force = Vector3(x=cmd_acc[0], y=cmd_acc[1], z=cmd_acc[2])
            msg.yaw_rate = cmd_acc[3]
            self.pub_cmd.publish(msg)

    def publish_viz(self):
        ## speed
        self.pub_speed.publish(Float32(data=np.linalg.norm(self.x0[7:10])))
        ## nmpc solving time
        self.pub_cpt.publish(Float32(data=self.nmpc.ocp.get_t()))
        ## neural sdf value
        self.pub_sdf.publish(Float32(data=self.nmpc.eval(0)[0]))

        if not self.failed:
            ## predicted traj
            msg = Path()
            msg.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.cfg.ros.frames.world)
            traj = self.nmpc.get_openloop_traj()
            for p, q in traj:
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position = Vector3(x=p[0], y=p[1], z=p[2])
                pose.pose.orientation = Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
                msg.poses.append(pose)
            self.pub_cmd_traj.publish(msg)

        ## command
        try:
            cmd_acc = self.nmpc.get_cmd_acc() if not self.failed else self.nmpc.cmd_acc_hover
            msg_viz = TwistStamped()
            msg_viz.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.cfg.ros.frames.body)
            msg_viz.twist.linear.x = cmd_acc[0]
            msg_viz.twist.linear.y = cmd_acc[1]
            msg_viz.twist.linear.z = cmd_acc[2]
            msg_viz.twist.angular.x = 0.0
            msg_viz.twist.angular.y = 0.0
            msg_viz.twist.angular.z = cmd_acc[3]
            self.pub_cmd_viz.publish(msg_viz)
        except:
            pass

    ## callbacks and services
    def cb_ref(self, msg):
        if self.ref is None:
            self.ref = Ref(self.cfg)
        self.t_ref = self.get_clock().now().nanoseconds / 1e9
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
            self.t_img = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            ts, xs = zip(*self.state_queue)  # transpose
            x = xs[np.argmin(np.abs(np.array(ts) - self.t_img))]
            self.nmpc.set_latent(msg.latent.data, x[:3], quat2rot(x[3:7]))
            # print("Latent received")

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
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        self.state_queue.appendleft((timestamp, self.x0.copy()))

    def srv_get_flag_callback(self, request, response):
        response.success = self.sdf_flag
        response.message = ''
        return response

    def srv_set_flag_callback(self, request, response):
        if request.data and not (self.nmpc.p[0, self.cfg.mpc.p_idx.W_p_Co]).any():
            self.get_logger().error('[sdf_nmpc] no image received, cannot activate constraints')
        else:
            self.sdf_flag = request.data
            print("SDF constraints set to:", self.sdf_flag)
        response.success = self.sdf_flag
        response.message = ''
        return response


## main
def main(args=None):
    np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
    
    rclpy.init(args=args)
    
    # Create temporary node to get parameters
    temp_node = Node('temp_param_node')
    temp_node.declare_parameter('cfg', '')
    temp_node.declare_parameter('rebuild', False)
    cfg_param = temp_node.get_parameter('cfg').value
    rebuild = temp_node.get_parameter('rebuild').value
    temp_node.destroy_node()
    
    cfg_file = f'params_{cfg_param}.yaml'

    if rebuild:
        path = os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file)

        print(f'[sdf_nmpc] building solver for {path}')
        build_solver(path)
        print(f'[sdf_nmpc] solver built')

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    ros_wrapper = RosWrapper(cfg)
    
    rclpy.spin(ros_wrapper)
    
    print('ROSPY DEAD. EXITING.')
    ros_wrapper.destroy_node()
    rclpy.shutdown()
    exit(0)


if __name__ == '__main__':
    main()
