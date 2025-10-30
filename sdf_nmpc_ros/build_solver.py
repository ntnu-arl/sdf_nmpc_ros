#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.ocp import build_solver


class BuildSolverNode(Node):
    def __init__(self):
        super().__init__('build_solver')
        
        # Declare and get parameter
        self.declare_parameter('cfg', '')
        cfg = self.get_parameter('cfg').value
        
        cfg_file = f'params_{cfg}.yaml'
        path = os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file)

        self.get_logger().info(f'building solver for {path}')
        build_solver(path)
        self.get_logger().info(f'solver built')


def main(args=None):
    rclpy.init(args=args)
    node = BuildSolverNode()
    rclpy.shutdown()
    exit(0)


if __name__ == '__main__':
    main()
