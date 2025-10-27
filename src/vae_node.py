import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.vae import VaeWrapper
import rospy
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import Image
from sdf_nmpc_ros.msg import Latent
from cv_bridge import CvBridge


class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('vae')

        self.cfg = cfg
        self.vae = VaeWrapper(cfg)

        self.bridge = CvBridge()

        topics = self.cfg.ros.topics
        self.pub_latent = rospy.Publisher(topics['latent'], Latent, tcp_nodelay=True, queue_size=1)
        self.pub_sdf = rospy.Publisher(topics.output['sdf_true'], Float32, tcp_nodelay=True, queue_size=1)
        self.sub_img = rospy.Subscriber(topics['obs'], Image, self.cb_img, tcp_nodelay=True, queue_size=1)
        ## debug topic
        self.pub_preproc = rospy.Publisher('/preprocessed_img', Image, tcp_nodelay=True, queue_size=1)

        rospy.loginfo('[vae] node started successfully')
        rospy.spin()

    def cb_img(self, msg):
        # img = np.ndarray((msg.height, msg.width), self.cfg.sensor.dtype, msg.data, 0)
        # img = np.ndarray((msg.height, msg.width), dtype=np.float32, buffer=msg.data, offset=0)
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        # print(img.dtype, np.min(img), np.max(img))
        # print(img)

        self.pub_sdf.publish(Float32(np.min(img)))

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
        msg_latent.latent = Float32MultiArray(data=latent.flatten())
        self.pub_latent.publish(msg_latent)


if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    RosWrapper(cfg)