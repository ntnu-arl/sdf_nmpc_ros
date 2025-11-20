#!/usr/bin/env python3
import os
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from sdf_nmpc import default_data_dir
from sdf_nmpc.utils.config import Config
from sdf_nmpc.utils.pos_sampler import PosSampler
from sdf_nmpc_ros.msg import Latent


class VizSdf3DNode(Node):
    def __init__(self):
        super().__init__('viz_sdf_2d')

        ## load config
        self.declare_parameter('cfg', 'default.yaml')
        cfg_file = self.get_parameter('cfg').get_parameter_value().string_value
        self.cfg = Config(cfg_file)

        self.declare_parameter('tol', 2.5e-2)
        self.declare_parameter('nb_points', 10000)
        self.tol = self.get_parameter('tol').get_parameter_value().double_value
        nb_points = self.get_parameter('nb_points').get_parameter_value().integer_value
        self.lvlset = self.cfg.robot.size.xy + self.cfg.mpc.bound_margin

        ## get sdf network, TODO should be an api call instead
        self.sdf = torch.jit.load(f'{default_data_dir()}/{self.cfg.nn.sdf_weights}')
        self.sdf.to(self.cfg.nn.vae_device)  # vae not sdf, since we want it on gpu is gpu is used
        self.sdf.eval()

        ## sample points in frustrum
        pos_sampler = PosSampler(
            self.cfg.sensor.dmax * 0.95,
            self.cfg.sensor.hfov, self.cfg.sensor.vfov,
            0, device=self.cfg.nn.vae_device
        )
        self.points = pos_sampler.grid_frustrum(nb_points, add_margin=False)
        self.nb_points = self.points.shape[0]

        ## topics
        self.pub_pc = self.create_publisher(PointCloud2, 'viz/sdf_lvlset', 1)
        self.sub_latent = self.create_subscription(Latent, 'latent', self.cb_latent, 1)

        self.get_logger().info('node started successfully')

    def pc_to_msg(self, ts, pc, norm=0.0, val=0.0):
        data = np.zeros([pc.shape[0], 5], dtype=np.float32)
        data[:, :3] = pc
        data[:, 3] = norm
        data[:, 4] = val
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='distance', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        header = Header(stamp=ts, frame_id=self.cfg.ros.frames.sensor)
        return point_cloud2.create_cloud(header, fields, data)

    def cb_latent(self, msg: Latent):
        with torch.no_grad():
            latent = torch.tensor(msg.latent.data, dtype=torch.float32, device=self.cfg.nn.vae_device).reshape(1, -1)
            points_sdf = self.sdf(torch.hstack([self.points, latent.repeat(self.nb_points, 1)])).flatten()

            ## get lvlset
            idx = (points_sdf - self.lvlset).abs() < self.tol
            pc_lvlset = self.points[idx]
            norms = torch.norm(pc_lvlset, dim=1)

            ## publish
            msg = self.pc_to_msg(msg.header.stamp, pc_lvlset.cpu().numpy(), norm=norms.cpu().numpy())
            self.pub_pc.publish(msg)


if __name__ == '__main__':
    rclpy.init()
    node = VizSdf3DNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
