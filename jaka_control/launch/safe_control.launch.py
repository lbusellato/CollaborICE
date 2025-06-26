import os
import launch
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    cbf_arch_arg = DeclareLaunchArgument(
        'cbf_arch', default_value='predictive', description='Which CBF architecture to use. Choices: [\'vanilla\', \'predictive\', \'sp\']')
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='False', description='Visualize everything in RVIZ')
    use_joint_gui_arg = DeclareLaunchArgument(
        'use_joint_gui', default_value='False', description='Use Joint State Publisher GUI')
    visualize_leap_arg = DeclareLaunchArgument(
        'visualize_leap', default_value='True', description='Visualize data from a connected Leap camera')
    simulate_robot_arg = DeclareLaunchArgument(
        'simulated_robot', default_value='False', description='Simulate the robot')
    forecasting_method_arg = DeclareLaunchArgument(
        'forecasting_method', default_value='nn', description='Forecasting method')
    
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

    # Launch the JAKA Safe Control node
    simulated_robot = LaunchConfiguration("simulated_robot")
    forecasting_method = LaunchConfiguration("forecasting_method")
    jaka_safe_control_node = Node(
        package='jaka_control',
        executable='safe_control',
        parameters=[{'simulated_robot': simulated_robot,
                     'forecasting_method' : forecasting_method}],
        output='screen',
        emulate_tty=True
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

    leap_streamer_launch = os.path.join(
        get_package_share_directory('leap_stream'),
        'launch',
        'leap_streamer.launch.py'
    )

    return launch.LaunchDescription([
        frame_id_arg,
        publish_rate_arg,
        tracking_mode_arg,
        cbf_arch_arg,
        use_joint_gui_arg,
        visualize_leap_arg,
        use_rviz_arg,
        jaka_display_launch,
        simulate_robot_arg,
        forecasting_method_arg,
        jaka_safe_control_node,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(leap_streamer_launch)
        )
    ])