import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.vae import VaeWrapper
import rospy
from std_msgs.msg import Header, Float32MultiArray, Float32
from sensor_msgs.msg import Image
from sdf_nmpc_ros.msg import Latent


class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('vae')

        self.cfg = cfg
        self.vae = VaeWrapper(cfg)

        topics = self.cfg.ros.topics
        self.sub_img = rospy.Subscriber(topics['obs'], Image, self.cb_img, tcp_nodelay=True, queue_size=1)
        self.pub_latent = rospy.Publisher(topics['latent'], Latent, tcp_nodelay=True, queue_size=1)
        self.pub_sdf = rospy.Publisher(topics.output['sdf_true'], Float32, tcp_nodelay=True, queue_size=1)

        rospy.loginfo('node vae started successfully')
        rospy.spin()

    def cb_img(self, msg):
        img = np.ndarray((msg.height, msg.width), self.cfg.sensor.dtype, msg.data, 0)

        self.pub_sdf.publish(Float32(np.min(img)))

        self.vae.set_img(img)
        latent = self.vae.encode()

        msg_latent = Latent()
        msg_latent.header = msg.header
        msg_latent.latent = Float32MultiArray(data=latent.flatten())
        self.pub_latent.publish(msg_latent)


if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    RosWrapper(cfg)
