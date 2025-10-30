#!/usr/bin/env python3

import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.vae import VaeWrapper
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
from sdf_nmpc_ros.msg import Latent
# from cv_bridge import CvBridge


class RosWrapper(Node):
    def __init__(self, cfg):
        super().__init__('vae')

        self.cfg = cfg
        self.vae = VaeWrapper(cfg)

        # self.bridge = CvBridge()

        topics = self.cfg.ros.topics
        self.pub_latent = self.create_publisher(Latent, topics['latent'], 1)
        self.pub_sdf = self.create_publisher(Float32, topics.output['sdf_true'], 1)
        self.sub_img = self.create_subscription(Image, topics['obs'], self.cb_img, 1)
        ## debug topic
        self.pub_preproc = self.create_publisher(Image, '/preprocessed_img', 1)

        self.get_logger().info('[vae] node started successfully')

    def cb_img(self, msg):
        img = np.ndarray((msg.height, msg.width), self.cfg.sensor.dtype, msg.data, 0)
        # img = np.ndarray((msg.height, msg.width), dtype=np.float32, buffer=msg.data, offset=0)
        # img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        # print(img.dtype, np.min(img), np.max(img))
        # print(img)

        # self.pub_sdf.publish(Float32(data=np.min(img)))

        self.vae.set_img(img)

        ## debug
        preproc_img = self.vae.img.cpu().numpy()
        msg_img = Image()
        msg_img.header = msg.header
        msg_img.height = self.cfg.sensor.shape_imgs[1]
        msg_img.width = self.cfg.sensor.shape_imgs[2]
        msg_img.encoding = 'mono8'
        msg_img.step = self.cfg.sensor.shape_imgs[2]
        msg_img.data = (preproc_img * 255).astype('uint8').tobytes()
        self.pub_preproc.publish(msg_img)
    
        latent = self.vae.encode()

        msg_latent = Latent()
        msg_latent.header = msg.header
        msg_latent.latent = Float32MultiArray(data=latent.flatten().tolist())
        self.pub_latent.publish(msg_latent)


def main(args=None):
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
