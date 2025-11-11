from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace, SetParameter, SetRemap
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    cfg = LaunchConfiguration('cfg')
    
    declare_args = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
    #     DeclareLaunchArgument('ns', default_value='sdf_nmpc', description='Common namespace for nmpc nodes'),
        DeclareLaunchArgument('cfg', default_value='sim_camera.yaml', description='Config preset <cfg>.yaml'),
    ]
    
    cfg_file = os.path.join(
        get_package_share_directory('sdf_nmpc_ros'),
        'config', 'sim_lidar.yaml'
    )
    if not cfg_file.endswith('.yaml'):
            cfg_file += '.yaml'

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

    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', os.path.join(get_package_share_directory('sdf_nmpc_ros'), 'rviz', 'nmpc.rviz')]
    )

    group = GroupAction([
        PushRosNamespace('/sdf_nmpc/'),  # this namespace is expected by rviz_nmpc_plugin
        # SetRemap(src='odometry', dst='/rmf/odom'),
        # SetRemap(src='observation', dst='/rmf/cam/depth'),
        # SetRemap(src='observation', dst='/rmf/lidar/range'),
        # SetRemap(src='cmd/acc', dst='/rmf/cmd/acc'),
        node_viz_vae,
        node_viz_sdf_2D,
        # node_viz_sdf_3D,
        node_rviz,
    ])
    
    return LaunchDescription(declare_args + [group])
