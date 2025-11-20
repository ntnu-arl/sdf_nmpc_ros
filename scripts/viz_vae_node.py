#!/usr/bin/env python3
import os
import numpy as np

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

from sdf_nmpc.utils.config import Config
from sdf_nmpc.vae import VaeWrapper
from sdf_nmpc_ros.msg import Latent


class VizVaeNode(Node):
    def __init__(self):
        super().__init__('viz_vae')

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
        
        self.pub_img = self.create_publisher(Image, 'viz/vae_reconstruction', 1)
        self.sub_latent = self.create_subscription(Latent, 'latent', self.cb_latent, sensor_qos)

        self.get_logger().info('node started successfully')

    def cb_latent(self, msg):
        latent = np.array(msg.latent.data, dtype='float32').reshape(1, -1)
        self.vae.set_latent(latent)
        img = self.vae.decode()

        msg_img = Image()
        msg_img.header = msg.header
        msg_img.height = self.cfg.sensor.shape_imgs[1]
        msg_img.width = self.cfg.sensor.shape_imgs[2]
        msg_img.encoding = 'mono8'
        msg_img.is_bigendian = 0
        msg_img.step = self.cfg.sensor.shape_imgs[2]
        msg_img.data = (img * 255).astype('uint8').tobytes()
        self.pub_img.publish(msg_img)


if __name__ == '__main__':
    rclpy.init()
    node = VizVaeNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
