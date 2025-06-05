import os
import numpy as np
import torch
import rospy
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
from collision_predictor_mpc.utils.visualization import Imgs2Points
from collision_predictor_mpc.utils import preprocessing
import ros_numpy
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2


class RosWrapper:
    def __init__(self, cfg, downsamp=4, outlier_rm=False):
        rospy.init_node('viz_pc')

        self.cfg = cfg

        self.preproc = torch.nn.Sequential(
            preprocessing.ToDevice(self.cfg.nn.vae_device),
            torch.jit.script(preprocessing.Reshape(self.cfg.sensor.shape_imgs)),
            torch.jit.script(preprocessing.ClipDistance(self.cfg.sensor.dmax, self.cfg.sensor.mm_resolution)),
            preprocessing.RemoveCloseOutliers(self.cfg.nn.vae_device) if outlier_rm else torch.nn.Identity(),
        )

        self.img_to_points = Imgs2Points(
            is_depth=self.cfg.sensor.is_depth,
            is_spherical=self.cfg.sensor.is_spherical,
            dmax=self.cfg.sensor.dmax,
            hfov=self.cfg.sensor.hfov,
            vfov=self.cfg.sensor.vfov,
            downsamp=downsamp,
            remove_d0=True,
            remove_dmax=True,
            device=self.cfg.nn.vae_device,
        )

        topics = self.cfg.ros.topics
        self.pub_pc = rospy.Publisher(f'{topics["obs"]}_pc', PointCloud2, tcp_nodelay=True, queue_size=1)
        self.pub_min = rospy.Publisher(f'{topics["obs"]}_min', Float32, tcp_nodelay=True, queue_size=1)
        self.sub_img = rospy.Subscriber(topics['obs'], Image, self.cb, tcp_nodelay=True, queue_size=1)

        rospy.spin()

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

        header = Header(stamp=ts, frame_id=self.cfg.ros.frames.body)
        return point_cloud2.create_cloud(header, fields, data)

    def cb(self, msg):
        img = np.ndarray((msg.height, msg.width), self.cfg.sensor.dtype, msg.data, 0)
        img = self.preproc(img)
        pc = self.img_to_points(img)[0]
        msg = self.pc_to_msg(msg.header.stamp, pc.cpu().numpy())
        self.pub_pc.publish(msg)
        self.pub_min.publish(Float32(min(torch.quantile(img[img > 0], 0.05).item() * self.cfg.sensor.dmax, 1)))



if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    RosWrapper(cfg)
