#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from collision_predictor_mpc.ocp import build_solver


if __name__ == '__main__':
    rclpy.init()
    node = Node('solver_builder')

    ## load config
    node.declare_parameter('cfg', 'default.yaml')
    cfg_file = node.get_parameter('cfg').get_parameter_value().string_value
    if not cfg_file.endswith('.yaml'):
        cfg_file += '.yaml'
    cfg_path = os.path.join(
        get_package_share_directory('sdf_nmpc_ros'),
        'config',
        cfg_path
    )
    node.get_logger().info(f'building solver for {cfg_path}')
    build_solver(cfg_path)
    node.get_logger().info('solver built')

    node.destroy_node()
    rclpy.shutdown()