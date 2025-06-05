from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    
    camera_name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='desk',
        description='Name to associate to the topic'
    )

    tracking_mode_arg = DeclareLaunchArgument(
        'tracking_mode',
        default_value='Desktop',
        description='Tracking mode (Desktop or HMD)'
    )

    leap_streamer_node = Node(
        package='leap_stream',
        executable='leap_streamer',
        name='leap_streamer',
        parameters=[{
            'camera_name': LaunchConfiguration('camera_name'),
            'tracking_mode': LaunchConfiguration('tracking_mode')
        }],
        output='screen'
    )

    return LaunchDescription([
        camera_name_arg,
        tracking_mode_arg,
        leap_streamer_node
    ])