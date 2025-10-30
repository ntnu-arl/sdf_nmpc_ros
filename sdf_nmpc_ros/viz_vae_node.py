#!/usr/bin/env python3

import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.vae import VaeWrapper

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sdf_nmpc_ros.msg import Latent


class RosWrapper(Node):
    def __init__(self, cfg):
        super().__init__('viz_vae')

        self.cfg = cfg
        self.vae = VaeWrapper(cfg)

        topics = self.cfg.ros.topics
        self.pub_img = self.create_publisher(Image, topics.viz['img_vae'], 1)
        self.sub_latent = self.create_subscription(
            Latent,
            topics['latent'],
            self.cb_latent,
            1
        )

        self.get_logger().info('[viz_vae] node started successfully')

    def cb_latent(self, msg: Latent):
        latent = np.array(msg.latent.data, dtype='float32').reshape(1, -1)
        self.vae.set_latent(latent)
        img = self.vae.decode()

        # Debug: Check for invalid values
        if np.isnan(img).any():
            self.get_logger().warn('[viz_vae] Image contains NaN values!')
            img = np.nan_to_num(img, nan=0.0)

        if np.isinf(img).any():
            self.get_logger().warn('[viz_vae] Image contains Inf values!')
            img = np.nan_to_num(img, posinf=1.0, neginf=0.0)

        # Debug: Print image statistics (with throttling)
        if not hasattr(self, '_last_log_time'):
            self._last_log_time = self.get_clock().now()

        now = self.get_clock().now()
        if (now - self._last_log_time).nanoseconds * 1e-9 > 1.0:
            self.get_logger().info(
                f'[viz_vae] Image stats - min: {img.min():.3f}, max: {img.max():.3f}, mean: {img.mean():.3f}'
            )
            self._last_log_time = now

        # Prepare Image message
        msg_img = Image()
        msg_img.header = msg.header
        msg_img.height = self.cfg.sensor.shape_imgs[1]
        msg_img.width = self.cfg.sensor.shape_imgs[2]
        msg_img.encoding = 'mono8'
        msg_img.step = self.cfg.sensor.shape_imgs[2]
        msg_img.data = (img * 255).astype('uint8').tobytes()

        self.pub_img.publish(msg_img)


def main(args=None):
    rclpy.init(args=args)

    # In ROS2, node parameters are managed differently.
    # You can either set this as a launch parameter or default to a YAML.
    node_name = 'viz_vae_param_loader'
    temp_node = rclpy.create_node(node_name)
    cfg_name = temp_node.declare_parameter('cfg', 'default').value
    temp_node.destroy_node()

    cfg_file = f'params_{cfg_name}.yaml'
    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    print("Loaded config from:", os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))

    node = RosWrapper(cfg)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
