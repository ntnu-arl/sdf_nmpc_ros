#!/usr/bin/env python3
import os
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image

from sdf_nmpc.utils.config import Config
from sdf_nmpc.vae import VaeWrapper
from sdf_nmpc_ros.msg import Latent


class VaeNode(Node):
    def __init__(self):
        super().__init__('vae')

        ## load config
        self.declare_parameter('cfg', 'default.yaml')
        cfg_file = self.get_parameter('cfg').get_parameter_value().string_value
        self.cfg = Config(cfg_file)
        self.vae = VaeWrapper(self.cfg)

        ## topics
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.pub_latent = self.create_publisher(Latent,'latent', 1)
        self.pub_sdf = self.create_publisher(Float32, 'output/sdf_true', 1)
        self.sub_img = self.create_subscription(Image, 'observation', self.cb_img, sensor_qos)

        self.get_logger().info('node started successfully')

    def cb_img(self, msg: Image):
        # img = np.frombuffer(msg.data, dtype=self.cfg.sensor.dtype)
        img = np.ndarray((msg.height, msg.width), self.cfg.sensor.dtype, msg.data, 0)

        ## publish true SDF, ie min over image
        self.pub_sdf.publish(Float32(data=float(np.min(img))))

        ## encode latent
        self.vae.set_img(img)
        latent = self.vae.encode()

        ## publish latent
        msg_latent = Latent()
        msg_latent.header = msg.header
        msg_latent.latent = Float32MultiArray()
        msg_latent.latent.data = latent.astype(np.float32).ravel().tolist()
        self.pub_latent.publish(msg_latent)


if __name__ == '__main__':
    rclpy.init()
    node = VaeNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()