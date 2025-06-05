import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    package_name = 'jaka_description'

    urdf_file_name = 'jaka.urdf.xacro'  # Change to your URDF file
    urdf_file = os.path.join(
        get_package_share_directory(package_name),
        'urdf',
        urdf_file_name)

    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    rviz_config_file = os.path.join(
        get_package_share_directory(package_name),
        'rviz',
        'playback.rviz'
    )

    return LaunchDescription([
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='real_robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_desc}],
            remappings=[
                ('/joint_states', '/joint_states')  # Real robot
            ]
        ),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='ghost_robot_state_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': False},
                {'tf_prefix': 'ghost_'}],
            remappings=[
                ('/joint_states', '/ghost_joint_states')  # Real robot
            ]
        ),

        Node(
            package='jaka_description',
            executable='log_playback_node',
            name='log_playback_node',
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_file]
        )
    ])