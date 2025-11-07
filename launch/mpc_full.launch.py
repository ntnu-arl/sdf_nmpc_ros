from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace, SetParameter, SetRemap
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # ns = LaunchConfiguration('ns')
    use_sim_time = LaunchConfiguration('use_sim_time')
    cfg = LaunchConfiguration('cfg')
    
    declare_args = [
        DeclareLaunchArgument('use_sim_time', default_value='false'),
    #     DeclareLaunchArgument('ns', default_value='sdf_nmpc', description='Common namespace for nmpc nodes'),
        DeclareLaunchArgument('cfg', default_value='magpie.yaml', description='Config preset <cfg>.yaml'),
    ]
    
    cfg_file = os.path.join(
        get_package_share_directory('sdf_nmpc_ros'),
        'config', 'magpie.yaml'
    )
    # if not cfg_file.endswith('.yaml'):
    #         cfg_file += '.yaml'

    node_vae = Node(
        package='sdf_nmpc_ros',
        executable='vae_node.py',
        name='vae',
        parameters=[{
            'cfg': cfg_file,
        }],
        output='screen'
    )

    node_ref_gen = Node(
        package='sdf_nmpc_ros',
        executable='ref_gen_node.py',
        name='ref_gen',
        parameters=[{
            'cfg': cfg_file,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    node_sdfnmpc = Node(
        package='sdf_nmpc_ros',
        executable='sdfnmpc_node.py',
        name='sdfnmpc',
        parameters=[{
            'cfg': cfg_file,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )
    

    node_viz_vae = Node(
        package='sdf_nmpc_ros',
        executable='viz_vae_node.py',
        name='viz_vae',
        parameters=[{
            'cfg': cfg_file,
            'use_sim_time': use_sim_time,
        }],
        output='screen'
    )

    node_viz_sdf_2D = Node(
        package='sdf_nmpc_ros',
        executable='viz_sdf_2D_node.py',
        name='viz_sdf_2D',
        parameters=[{
            'cfg': cfg_file,
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )

    node_viz_sdf_3D = Node(
        package='sdf_nmpc_ros',
        executable='viz_sdf_3D_node.py',
        name='viz_sdf_3D',
        parameters=[{
            'cfg': cfg_file,
            'use_sim_time': use_sim_time
        }],
        output='screen'
    )

    group = GroupAction([
        PushRosNamespace('/sdf_nmpc/'),  # this namespace is expected by rviz_nmpc_plugin
        # SetRemap(src='odometry', dst='/msf_core/odometry_50hz'),
        SetRemap(src='odometry', dst='/msf_core/odometry'),
        SetRemap(src='observation', dst='/img_node/range_image'),
        SetRemap(src='cmd/acc', dst='/mavros/setpoint_raw/local'),
        SetRemap(src='wps', dst='/gbplanner_path'),
        node_vae,
        node_ref_gen,
        node_sdfnmpc,
        # node_viz_vae,
        # node_viz_sdf_2D
        # node_viz_sdf_3D,
    ])
    
    return LaunchDescription(declare_args + [group])
