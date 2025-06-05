from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Launch arguments
    frame_id_arg = DeclareLaunchArgument(
        'frame_id',
        default_value='leap_motion',
        description='Frame ID for LeapMotion data'
    )
    
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate_limit',
        default_value='120.0',
        description='Maximum publish rate in Hz'
    )
    
    tracking_mode_arg = DeclareLaunchArgument(
        'tracking_mode',
        default_value='Desktop',
        description='Tracking mode (Desktop or HMD)'
    )

    # LeapMotion streamer node
    leap_streamer_node = Node(
        package='collaborice_leap_streamer',
        executable='leap_streamer',
        name='leap_motion_streamer',
        parameters=[{
            'frame_id': LaunchConfiguration('frame_id'),
            'publish_rate_limit': LaunchConfiguration('publish_rate_limit'),
            'tracking_mode': LaunchConfiguration('tracking_mode')
        }],
        output='screen'
    )

    return LaunchDescription([
        frame_id_arg,
        publish_rate_arg,
        tracking_mode_arg,
        leap_streamer_node
    ])