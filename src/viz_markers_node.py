import os
import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.utils.config import Config
import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import MultiDOFJointTrajectory
from geometry_msgs.msg import Vector3


class RosWrapper:
    def __init__(self, cfg):
        rospy.init_node('vae')

        self.cfg = cfg

        self.state = None

        topics = self.cfg.ros.topics
        self.pub_body = rospy.Publisher(topics.viz.markers['body'], Marker, tcp_nodelay=False, queue_size=1)
        self.pub_col = rospy.Publisher(topics.viz.markers['col'], Marker, tcp_nodelay=False, queue_size=1)
        self.pub_wp = rospy.Publisher(topics.viz.markers['wp'], Marker, tcp_nodelay=False, queue_size=1)
        self.pub_dir = rospy.Publisher(topics.viz.markers['dir'], Marker, tcp_nodelay=False, queue_size=1)
        self.sub_state = rospy.Subscriber(topics['odom'], Odometry, self.cb_state, tcp_nodelay=True, queue_size=1)
        self.sub_wps = rospy.Subscriber(self.cfg.ros.topics['ref_wps'], Path, self.cb_wps, tcp_nodelay=True, queue_size=1)
        self.sub_ref = rospy.Subscriber(self.cfg.ros.topics['ref_horizon'], MultiDOFJointTrajectory, self.cb_horizon, tcp_nodelay=True, queue_size=1)

        rospy.loginfo('node viz_markers started successfully')
        rospy.spin()

    def cb_state(self, msg):
        self.state = msg.pose.pose

        msg_marker = Marker()
        msg_marker.header = Header(stamp=msg.header.stamp, frame_id=self.cfg.ros.frames.world)
        msg_marker.type = Marker.CYLINDER
        msg_marker.pose = msg.pose.pose
        msg_marker.scale.x = self.cfg.robot.size.xy * 2
        msg_marker.scale.y = self.cfg.robot.size.xy * 2
        msg_marker.scale.z = self.cfg.robot.size.z * 2
        msg_marker.color.a = 1.
        msg_marker.color.r = 0.69
        msg_marker.color.g = 0.35
        msg_marker.color.b = 0.0
        self.pub_body.publish(msg_marker)

        msg_marker = Marker()
        msg_marker.header = Header(stamp=msg.header.stamp, frame_id=self.cfg.ros.frames.world)
        msg_marker.type = Marker.SPHERE
        msg_marker.pose = msg.pose.pose
        msg_marker.scale.x = (self.cfg.robot.size.xy + self.cfg.mpc.bound_margin) * 2
        msg_marker.scale.y = (self.cfg.robot.size.xy + self.cfg.mpc.bound_margin) * 2
        msg_marker.scale.z = (self.cfg.robot.size.xy + self.cfg.mpc.bound_margin) * 2
        msg_marker.color.a = 0.5
        msg_marker.color.r = 0.69
        msg_marker.color.g = 0.35
        msg_marker.color.b = 0.0
        self.pub_col.publish(msg_marker)

    def cb_wps(self, msg):
        msg_marker = Marker()
        msg_marker.header = Header(stamp=msg.header.stamp, frame_id=self.cfg.ros.frames.world)
        msg_marker.type = Marker.SPHERE
        msg_marker.pose.position = msg.poses[0].pose.position
        msg_marker.pose.orientation.w = 1
        msg_marker.scale.x = 0.2
        msg_marker.scale.y = 0.2
        msg_marker.scale.z = 0.2
        msg_marker.color.a = 1.0
        msg_marker.color.r = 1.0
        msg_marker.color.g = 1.0
        msg_marker.color.b = 1.0
        self.pub_wp.publish(msg_marker)

    def cb_horizon(self, msg):
        if self.state is None:
            return

        ## direction to goal
        msg_marker = Marker()
        msg_marker.header = Header(stamp=msg.header.stamp, frame_id=self.cfg.ros.frames.world)

        dir = np.array([
            msg.points[-1].transforms[0].translation.x - self.state.position.x,
            msg.points[-1].transforms[0].translation.y - self.state.position.y,
            msg.points[-1].transforms[0].translation.z - self.state.position.z,
        ])
        norm = np.linalg.norm(dir)

        if norm < 0.2:
            msg_marker.action = Marker.DELETE
            self.pub_dir.publish(msg_marker)
        elif norm > 1:
            dir = dir / norm
        dir += np.array([self.state.position.x, self.state.position.y, self.state.position.z])
        msg_marker.type = Marker.ARROW
        msg_marker.points.append(self.state.position)
        msg_marker.points.append(Vector3(*dir))
        msg_marker.scale.x = 0.07  # shaft diameter
        msg_marker.scale.y = 0.12  # head diameter
        msg_marker.scale.z = 0.07  # head length
        msg_marker.color.a = 1.0
        msg_marker.color.r = 1.0
        msg_marker.color.g = 1.0
        msg_marker.color.b = 1.0
        self.pub_dir.publish(msg_marker)


if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'

    cfg = Config(os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file))
    RosWrapper(cfg)
