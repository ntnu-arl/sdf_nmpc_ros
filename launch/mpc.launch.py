from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    # Declare launch arguments
    cfg_arg = DeclareLaunchArgument(
        'cfg',
        default_value='gbplanner_rotors',
        description='Configuration name'
    )
    
    rebuild_arg = DeclareLaunchArgument(
        'rebuild',
        default_value='false',
        description='Rebuild solver flag'
    )
    
    vae_arg = DeclareLaunchArgument(
        'vae',
        default_value='true',
        description='Enable VAE node'
    )

    viz_vae_arg = DeclareLaunchArgument(
        'viz_vae',
        default_value='true',
        description='Enable viz VAE node'
    )
    
    # Get launch configuration values
    cfg = LaunchConfiguration('cfg')
    rebuild = LaunchConfiguration('rebuild')
    vae = LaunchConfiguration('vae')
    viz_vae = LaunchConfiguration('viz_vae')
    
    # SDF NMPC node
    sdfnmpc_node = Node(
        package='sdf_nmpc_ros',
        executable='sdfnmpc_node.py',  # Changed: added .py
        name='sdfnmpc_node',
        output='screen',
        parameters=[
            {'cfg': cfg},
            {'rebuild': rebuild}
        ],
        prefix='taskset -c 0-1'
    )
    
    # Reference generator node
    ref_gen_node = Node(
        package='sdf_nmpc_ros',
        executable='ref_gen_node.py',  # Changed: added .py
        name='ref_gen_node',
        output='screen',
        parameters=[
            {'cfg': cfg},
            {'rebuild': rebuild}
        ],
        prefix='taskset -c 2'
    )
    
    # VAE node (conditional)
    vae_node = Node(
        package='sdf_nmpc_ros',
        executable='vae_node.py',  # Changed: added .py
        name='vae_node',
        output='screen',
        parameters=[
            {'cfg': cfg},
            {'rebuild': rebuild}
        ],
        prefix='taskset -c 3',
        condition=IfCondition(vae)
    )

    # Viz VAE node (conditional)
    viz_vae_node = Node(
        package='sdf_nmpc_ros',
        executable='viz_vae_node.py',  # Changed: added .py
        name='viz_vae_node',
        output='screen',
        parameters=[
            {'cfg': cfg}
        ],
        prefix='taskset -c 3',
        condition=IfCondition(viz_vae)
    )
    
    return LaunchDescription([
        cfg_arg,
        rebuild_arg,
        vae_arg,
        viz_vae_arg,
        sdfnmpc_node,
        ref_gen_node,
        vae_node,
        viz_vae_node
    ])