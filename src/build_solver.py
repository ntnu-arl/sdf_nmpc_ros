import os
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR
from collision_predictor_mpc.gen_model import build


if __name__ == '__main__':
    cfg_file = f'params_{rospy.get_param('/cfg_file')}.yaml'
    path = os.path.join(COLPREDMPC_CONFIG_DIR, cfg_file)
    rospy.loginfo(f'building solver for {args.mode}, loading {path}')
    build(path)
    rospy.loginfo(f'solver built')
    exit(0)
