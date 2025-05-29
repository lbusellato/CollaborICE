import rclpy
from rclpy.node import Node
import json
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from jaka_interface.pose_conversions import leap_to_jaka
from geometry_msgs.msg import Point
import numpy as np

class LeapMotionVisualizer(Node):
    def __init__(self):
        super().__init__('leap_motion_visualizer_node')

        # Subscribe to Leap Motion JSON topic
        self.subscription = self.create_subscription(
            String, 
            '/sensors/leapFusion/json', 
            self.leap_callback, 
            10
        )
        self.subscription_ = self.create_subscription(
            String,     
            '/applications/hand_forecasting/kalman', 
            self.forecasting_callback, 
            10
        )

        # Publisher for visualizing hand joints as spheres in RViz
        self.marker_publisher = self.create_publisher(MarkerArray, '/leap/visualization/hand_markers', 10)
        self.last_marker_count = 0  # Track last marker count

        self.fmarker_publisher = self.create_publisher(MarkerArray, '/forecasting/hand_markers', 10)
        self.flast_marker_count = 0  # Track last marker count

        self.min_hand_confidence = 0.1

    def forecasting_callback(self, msg):

        try:
            forecasting_data = json.loads(msg.data)
            future_position = forecasting_data.get('future_position', [])
            
            if not future_position:
                self.clear_markers(self.fmarker_publisher)
                return  

            marker_array = MarkerArray()

            marker_id = 0
            self.add_marker(marker_array, marker_id, future_position  , "forecast", "", scale=0.05, rgb=[1,0,0])

            self.fmarker_publisher.publish(marker_array)
        
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse Forecasting JSON data.")

    def leap_callback(self, msg):
        try:
            leap_data = json.loads(msg.data)  # Parse JSON
            hands = leap_data.get("hands", [])  # Extract hand data
                       
            marker_array = MarkerArray()

            if not hands:
                self.clear_markers(self.marker_publisher)
                return  

            marker_id = 0
            self.add_marker(marker_array, marker_id, [0,0,0], "leap", "", 0.05)
            marker_id += 1
                        
            for hand in hands:
                if hand is None:
                    continue
                # Filter out low-confidence hand detections
                if hand.get('confidence') > self.min_hand_confidence:
                    hand_type = hand.get("hand_type", "unknown")
                    keypoints = hand.get("hand_keypoints", {})

                    # Palm Position
                    palm_pos = keypoints.get("palm_position", None)
                    if palm_pos:
                        self.add_marker(marker_array, marker_id, palm_pos, "palm", hand_type)
                        marker_id += 1

                    # Finger Joints
                    fingers = keypoints.get("fingers", {})
                    max_dist = 0           
                    for finger_name, joints in fingers.items():
                        for joint_name, pos in joints.items():
                            # Each finger is defined by two joints
                            self.add_marker(marker_array, marker_id, pos['prev_joint'], f"{finger_name}_{joint_name}", hand_type)
                            marker_id += 1
                            dist = np.linalg.norm(np.abs(np.array(pos['prev_joint']) - np.array(palm_pos)))
                            if dist > max_dist:
                                max_dist = dist
                
                    self.add_marker(marker_array, marker_id, palm_pos, "palm", hand_type, max_dist * 2, 0.3)
                    marker_id += 1
            self.last_marker_count = marker_id  # Track how many markers were added
            self.marker_publisher.publish(marker_array)

        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse Leap Motion JSON data.")

    def add_marker(self, marker_array, marker_id, pos, joint_name, hand_type, scale=0.02, alpha=1.0, rgb=None):
        """Creates and adds a sphere marker for RViz visualization."""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "leap_hand_joints"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        pos = leap_to_jaka(pos)/1000
        marker.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale  # Sphere size (2 cm)
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.a = alpha  # Fully visible
        if rgb is None:
            r, g, b = [0.0, 0.0 if hand_type == "right" else 1.0, 1.0 if hand_type == "right" else 0.0]
        else:
            r, g, b = rgb
        marker.color.r = float(r)
        marker.color.g =  float(g)
        marker.color.b =  float(b)

        marker_array.markers.append(marker)

    def clear_markers(self, mp):
        """Clears all markers from RViz when no hand is detected."""
        marker_array_msg = MarkerArray()
        marker = Marker()
        marker.id = 0
        marker.ns = "leap_hand_joints"
        marker.action = Marker.DELETEALL
        marker_array_msg.markers.append(marker)
        mp.publish(marker_array_msg)

def main():
    rclpy.init()
    node = LeapMotionVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
