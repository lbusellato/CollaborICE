import rclpy
import json
import numpy as np

from jaka_messages.msg import LeapHand
from jaka_interface.pose_conversions import leap_to_jaka
from rclpy.node import Node
from std_msgs.msg import String

class LeapSubscriberNode(Node):
    def __init__(self):
        super().__init__('leap_subscriber_node')

        self.declare_parameter('min_hand_confidence', 0.33)
        self.min_hand_confidence = self.get_parameter('min_hand_confidence').value
        
        self.create_subscription(String, '/sensors/leap/json', self.leap_subscriber_callback, qos_profile=1)

        self.leap_publisher = self.create_publisher(LeapHand, '/jaka/control/hand', qos_profile=1)
        leap_freq = 120 # Hz
        leap_dt = 1 / leap_freq
        self.leap_publisher_timer = self.create_timer(leap_dt, self.leap_publisher_callback)

        self.hand_pos = None
        self.hand_radius = 0
    
    def leap_subscriber_callback(self, msg: String):
        data = json.loads(msg.data)
        hands = data.get('hands')
        flag = True
        if hands:
            # Filter out low-confidence hand detections
            confidences = np.array([h.get('confidence') for h in hands])
            hand = hands[np.argmax(confidences)]
            if hand.get('confidence') > self.min_hand_confidence:
                keypoints = hand.get('hand_keypoints')

                fingers = keypoints.get("fingers", {})
                palm_position = np.array(keypoints.get('palm_position'))

                joint_positions = np.array([joint_pos.get('prev_joint') for _, finger in fingers.items() 
                                                                        for _, joint_pos in finger.items()])
                if len(joint_positions) > 0:
                    # Compute a bounding sphere from the palm to the farthest away hand joint
                    dists = np.linalg.norm(joint_positions - palm_position, axis=1)
                    # Convert to JAKA world
                    self.hand_radius = np.max(dists) * 1000
                    self.hand_pos = leap_to_jaka(palm_position)
                    flag = False
        if flag: # No hand was detected
            self.hand_radius = 0
            self.hand_pos = None

    def leap_publisher_callback(self):
        if self.hand_pos is not None:
            x, y, z, = [*self.hand_pos]
            self.leap_publisher.publish(LeapHand(x=x, y=y, z=z, radius=self.hand_radius))
def main():
    rclpy.init()
    node = LeapSubscriberNode()    
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()

if __name__=='__main__':
    main()