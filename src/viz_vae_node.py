# import os
# import numpy as np
# from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
# from collision_predictor_mpc.utils.config import Config
# from collision_predictor_mpc.vae import VaeWrapper
# import rospy
# from sensor_msgs.msg import Image
# from sdf_nmpc_ros.msg import Latent


# class RosWrapper:
#     def __init__(self, cfg):
#         rospy.init_node('viz_vae')

#         self.cfg = cfg
#         self.vae = VaeWrapper(cfg)

#         topics = self.cfg.ros.topics
#         self.pub_img = rospy.Publisher(topics.viz['img_vae'], Image, tcp_nodelay=True, queue_size=1)
#         self.sub_latent = rospy.Subscriber(topics['latent'], Latent, self.cb_latent, tcp_nodelay=True, queue_size=1)

#         rospy.loginfo('[viz_vae] node started successfully')
#         rospy.spin()

#     def cb_latent(self, msg):
#         latent = np.array(msg.latent.data, dtype='float32').reshape(1,-1)
#         self.vae.set_latent(latent)
#         img = self.vae.decode()

#         msg_img = Image()
#         msg_img.header = msg.header
#         msg_img.height = self.cfg.sensor.shape_imgs[1]
#         msg_img.width = self.cfg.sensor.shape_imgs[2]
#         msg_img.encoding = '8UC1'
#         msg_img.step = self.cfg.sensor.shape_imgs[2]
#         msg_img.data = (img * 255).astype('uint8').tobytes()

#         self.pub_img.publish(msg_img)


# if __name__ == '__main__':
#     cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

#     cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
#     RosWrapper(cfg)


import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.vae import VaeWrapper
import rospy
from sensor_msgs.msg import Image
from sdf_nmpc_ros.msg import Latent


class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('viz_vae')

        self.cfg = cfg
        self.vae = VaeWrapper(cfg)

        topics = self.cfg.ros.topics
        self.pub_img = rospy.Publisher(topics.viz['img_vae'], Image, tcp_nodelay=True, queue_size=1)
        self.sub_latent = rospy.Subscriber(topics['latent'], Latent, self.cb_latent, tcp_nodelay=True, queue_size=1)

        rospy.loginfo('[viz_vae] node started successfully')
        rospy.spin()

    def cb_latent(self, msg):
        latent = np.array(msg.latent.data, dtype='float32').reshape(1,-1)
        self.vae.set_latent(latent)
        img = self.vae.decode()

        # Debug: Check for invalid values
        if np.isnan(img).any():
            rospy.logwarn('[viz_vae] Image contains NaN values!')
            img = np.nan_to_num(img, nan=0.0)
        
        if np.isinf(img).any():
            rospy.logwarn('[viz_vae] Image contains Inf values!')
            img = np.nan_to_num(img, posinf=1.0, neginf=0.0)
        
        # Clip values to valid range [0, 1] before scaling
        # img = np.clip(img, 0.0, 1.0)
        
        # Debug: Print image statistics
        rospy.loginfo_throttle(1.0, f'[viz_vae] Image stats - min: {img.min():.3f}, max: {img.max():.3f}, mean: {img.mean():.3f}')

        msg_img = Image()
        msg_img.header = msg.header
        msg_img.height = self.cfg.sensor.shape_imgs[1]
        msg_img.width = self.cfg.sensor.shape_imgs[2]
        msg_img.encoding = 'mono8'  # Changed from '8UC1' to standard 'mono8'
        msg_img.step = self.cfg.sensor.shape_imgs[2]
        msg_img.data = (img * 255).astype('uint8').tobytes()

        self.pub_img.publish(msg_img)


if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    print("Loaded config from:", os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    RosWrapper(cfg)