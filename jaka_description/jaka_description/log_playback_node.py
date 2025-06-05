import rclpy
import json
import time
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Point

class LogPlaybackNode(Node):
    def __init__(self):
        super().__init__('log_playback_node')

        # Publishers
        self.joint_state_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.ghost_joint_state_pub = self.create_publisher(JointState, '/ghost_joint_states', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/tcp_trajectory', 10)

        # Load the execution log
        self.log_data = self.load_log_file('/home/realsense/ros2_ws/install/jaka_control/share/jaka_control/logs/execution_log-1741177543.9149256.json')
        self.current_index = 0

        # Timer for replay
        self.timer = self.create_timer(0.008, self.replay_log)  # 20Hz playback
        self.start_time = time.time()
        self.first_timestamp = self.log_data[0]['timestamp'] if self.log_data else 0

    def load_log_file(self, filepath):
        try:
            with open(filepath, 'r') as f:
                return [json.loads(line) for line in f]
        except Exception as e:
            self.get_logger().error(f"Failed to load log file: {e}")
            return []

    def replay_log(self):
        if self.current_index >= len(self.log_data):
            self.get_logger().info("Log playback complete.")
            self.timer.cancel()
            return

        entry = self.log_data[self.current_index]
        elapsed_time = time.time() - self.start_time
        log_time = entry['timestamp'] - self.first_timestamp

        if elapsed_time >= log_time:
            self.publish_joint_states(entry)
            self.publish_tcp_marker(entry)
            self.current_index += 1

    def publish_joint_states(self, entry):
        joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

        # Actual joint state
        joint_msg = JointState()
        joint_msg.header = Header(stamp=self.get_clock().now().to_msg())
        joint_msg.name = joint_names
        joint_msg.position = entry['joint_corrected']
        self.joint_state_pub.publish(joint_msg)

        # Ghost (Nominal) joint state
        ghost_joint_msg = JointState()
        ghost_joint_msg.header = Header(stamp=self.get_clock().now().to_msg())
        ghost_joint_msg.name = joint_names
        ghost_joint_msg.position = entry['joint_nominal']
        self.ghost_joint_state_pub.publish(ghost_joint_msg)

    def publish_tcp_marker(self, entry):
        marker_array = MarkerArray()
        tcp_position = entry['tcp_position']

        # Actual trajectory marker
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "tcp_actual"
        marker.id = self.current_index
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=tcp_position[0], y=tcp_position[1], z=tcp_position[2])
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0  # Green for actual
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker_array.markers.append(marker)

        # Ghost trajectory marker
        ghost_marker = Marker()
        ghost_marker.header.frame_id = "world"
        ghost_marker.header.stamp = self.get_clock().now().to_msg()
        ghost_marker.ns = "tcp_nominal"
        ghost_marker.id = self.current_index + 10000
        ghost_marker.type = Marker.SPHERE
        ghost_marker.action = Marker.ADD
        ghost_marker.pose.position = Point(x=tcp_position[0] + 0.01, y=tcp_position[1] + 0.01, z=tcp_position[2] + 0.01)
        ghost_marker.scale.x = ghost_marker.scale.y = ghost_marker.scale.z = 0.05
        ghost_marker.color.r = 1.0  # Red for nominal
        ghost_marker.color.g = 0.0
        ghost_marker.color.b = 0.0
        ghost_marker.color.a = 0.5  # Transparent
        marker_array.markers.append(ghost_marker)

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = LogPlaybackNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()