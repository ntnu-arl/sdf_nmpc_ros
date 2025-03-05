import os
import rospy
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.ocp import build_solver


if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param("/cfg")}.yaml'
    path = os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file)

    rospy.loginfo(f'building solver for {path}')
    build_solver(path)
    rospy.loginfo(f'solver built')
    exit(0)
