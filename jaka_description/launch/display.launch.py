import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    package_name = 'jaka_description'

    # Define launch arguments
    use_joint_gui_arg = DeclareLaunchArgument(
        'use_joint_gui', default_value='False', description='Use Joint State Publisher GUI')
    visualize_leap_arg = DeclareLaunchArgument(
        'visualize_leap', default_value='True', description='Visualize data from a connected Leap camera')

    urdf_file_name = 'jaka.urdf'  # Change to your URDF file
    urdf_file = os.path.join(
        get_package_share_directory(package_name),
        'urdf',
        urdf_file_name)

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    rviz_config_file = os.path.join(
        get_package_share_directory(package_name),
        'rviz',
        'display.rviz'
    )

    use_joint_gui = LaunchConfiguration("use_joint_gui")
    # Joint state publisher nodes
    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        condition=IfCondition(
            PythonExpression([
                use_joint_gui,
                ' == True',
            ]))
    )

    visualize_leap = LaunchConfiguration("visualize_leap")
    leap_visualizer_node = Node(
        package='jaka_description',
        executable='leap_visualizer_node',
        name='leap_visualizer_node',
        condition=IfCondition(
            PythonExpression([
                visualize_leap,
                ' == True',
            ]))
    )

    return LaunchDescription([
        use_joint_gui_arg,
        visualize_leap_arg,
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}]
        ),

        joint_state_publisher_gui_node,
        leap_visualizer_node,

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        )
    ])