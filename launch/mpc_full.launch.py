from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch_ros.actions import Node, PushRosNamespace, SetParameter, SetRemap
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    ns = LaunchConfiguration('ns')
    # cfg = LaunchConfiguration('cfg')
    # build = LaunchConfiguration('build')

    declare_args = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('ns', default_value='sdf_nmpc', description='Common namespace for nmpc nodes'),
        # DeclareLaunchArgument('cfg', default_value='sim_camera.yaml', description='Config preset <cfg>.yaml'),
        # # DeclareLaunchArgument('build', default_value='false', description='Run pre-build step before starting nodes'),
    ]

    cfg_file = PathJoinSubstitution([
        get_package_share_directory('sdf_nmpc_ros'),
        'config',
        'sim_lidar.yaml'
    ])

    node_vae = Node(
        package='sdf_nmpc_ros',
        executable='vae_node.py',
        name='vae',
        parameters=[{
            'cfg': cfg_file,
            'use_sim_time': use_sim_time,
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

    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', os.path.join(get_package_share_directory('sdf_nmpc_ros'), 'rviz', 'nmpc.rviz')]
    )

    group = GroupAction([
        PushRosNamespace(ns),  # this namespace is expected by rviz_nmpc_plugin
        SetRemap(src='odometry', dst='/rmf/odom'),
        # SetRemap(src='observation', dst='/rmf/cam/depth'),
        SetRemap(src='observation', dst='/rmf/lidar/range'),
        SetRemap(src='cmd/acc', dst='/rmf/cmd/acc'),
        node_vae,
        node_ref_gen,
        node_sdfnmpc,
        node_rviz,
        node_viz_vae,
        node_viz_sdf_2D,
        node_viz_sdf_3D,
    ])
    
    return LaunchDescription(declare_args + [group])
