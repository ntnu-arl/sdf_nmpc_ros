import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.vae import VaeWrapper
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header, Float32MultiArray
import os


class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('vae')

        self.cfg = cfg
        self.vae = VaeWrapper(cfg)

        topics = self.cfg.ros.topics
        self.sub_img = rospy.Subscriber(topics['img_input'], Image, self.cb_img, tcp_nodelay=True, queue_size=1)
        self.pub_img = rospy.Publisher(topics['img_output'], Image, tcp_nodelay=True, queue_size=1)
        self.pub_latent = rospy.Publisher(topics['latent'], Float32MultiArray, tcp_nodelay=True, queue_size=1)

        rospy.spin()

    def cb_img(self, msg):
        img = np.ndarray((msg.height, msg.width), self.cfg.sensor.dtype, msg.data, 0)
        self.vae.set_img(img)
        latent = self.vae.encode()

        msg_latent = Float32MultiArray()
        msg_latent.data = latent.flatten()
        self.pub_latent.publish(msg_latent)

        msg_img = Image()
        msg_img.header = msg.header
        msg_img.height = msg.height
        msg_img.width = msg.width
        msg_img.encoding = '8UC1'
        msg_img.step = msg.width
        msg_img.data = (self.vae.decode() * 255).astype('uint8').tobytes()

        self.pub_img.publish(msg_img)


if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param("/cfg_file")}.yaml'

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    RosWrapper(cfg)
