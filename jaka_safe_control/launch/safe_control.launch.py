import os
import launch
import launch_ros.actions
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    cbf_arch_arg = DeclareLaunchArgument(
        'cbf_arch', default_value='vanilla', description='Which CBF architecture to use. Choices: [\'vanilla\', \'predictive\', \'sp\']')
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='False', description='Visualize everything in RVIZ')
    use_joint_gui_arg = DeclareLaunchArgument(
        'use_joint_gui', default_value='False', description='Use Joint State Publisher GUI')
    visualize_leap_arg = DeclareLaunchArgument(
        'visualize_leap', default_value='True', description='Visualize data from a connected Leap camera')
    
    # Get package directory
    jaka_description = get_package_share_directory('jaka_description')

    # Include the existing JakaInterface launch file, passing the arguments
    use_rviz = LaunchConfiguration("use_rviz")
    jaka_display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(jaka_description, 'launch', 'display.launch.py')),
        launch_arguments={
            'use_joint_gui': LaunchConfiguration('use_joint_gui'),
            'visualize_leap': LaunchConfiguration('visualize_leap'),
        }.items(),
        condition=IfCondition(
            PythonExpression([
                use_rviz,
                ' == True',
            ]))
    )

    # Launch the Leap subscriber node
    leap_subscriber_node = Node(
        package='jaka_safe_control',
        executable='leap_subscriber_node',
        output='screen'
    )

    # Launch the JAKA Safe Control node
    jaka_safe_control_node = Node(
        package='jaka_safe_control',
        executable=[LaunchConfiguration('cbf_arch'), '_triangle_wave'],
        output='screen',
        parameters=[{'publish_robot_state': LaunchConfiguration('use_rviz')}]
    )

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

    return launch.LaunchDescription([
        frame_id_arg,
        publish_rate_arg,
        tracking_mode_arg,
        leap_streamer_node,
        leap_subscriber_node,
        cbf_arch_arg,
        use_joint_gui_arg,
        visualize_leap_arg,
        use_rviz_arg,
        jaka_display_launch,
        jaka_safe_control_node,
    ])
