import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
from std_msgs.msg import String

import json

from collaborice_leap_streamer.lib_fusion import separate_hands, fuse_hand


class LeapFusion(Node):

    def __init__(self):
        super().__init__('leap_fusion')
        self.subscription1 = self.create_subscription(String, '/sensors/leapDesk/json', self.listener_callback_1, 1)
        self.subscription2 = self.create_subscription(String, '/sensors/leapScreen/json', self.listener_callback_2, 1)

        self.hand_left_leap1 = None
        self.hand_right_leap1 = None
        self.last_update_leap1 = 0

        self.hand_left_leap2 = None
        self.hand_right_leap2 = None
        self.last_update_leap2 = 0

        self.time_passed = 1000
        self.publisher_ = self.create_publisher(String, '/sensors/leapFusion/json', 1)
        #self.timer = self.create_timer(0.1, self.publish_joints)  # Timer for publishing data

        self.frame_counter = 0

    def listener_callback_1(self, msg):
        print('call desktop')
        new_data = msg.data
        try:
            # Parse the JSON formatted string into a dictionary
            data = json.loads(new_data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse leap data: {e}")
            # self.latest_leap_data = None
            return

        this_time = data['timestamp']

        if this_time < self.last_update_leap1:
            return
        #print(this_time)
        self.last_update_leap1 = this_time
        hands = data['hands']
        left_hand, right_hand = separate_hands(hands)
        self.hand_left_leap1 = left_hand
        self.hand_right_leap1 = right_hand
        self.fuse_data()

    def listener_callback_2(self, msg):
        print('call screen')
        new_data = msg.data
        try:
            # Parse the JSON formatted string into a dictionary
            data = json.loads(new_data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Failed to parse leap data: {e}")
            # self.latest_leap_data = None
            return

        this_time = data['timestamp']
        if this_time < self.last_update_leap2:
            return
        self.last_update_leap2 = this_time
        hands = data['hands']
        left_hand, right_hand = separate_hands(hands)
        self.hand_left_leap2 = left_hand
        self.hand_right_leap2 = right_hand
        self.fuse_data()

    def publish_joints(self):
        pass  # The listener handles publishing data

    def create_frame(self, fused_left_hand, fused_right_hand, current_time):
        frame_data = {
            'frame_id': self.frame_counter,
            'timestamp': current_time,
            'hands': [fused_left_hand, fused_right_hand]
        }
        json_output = json.dumps(frame_data)

        # Publish JSON
        msg = String()
        msg.data = json_output
        return msg

    def fuse_data(self):
        """ Fuse the hand data from both Leap Motion devices """
        current_time = time.time()

        '''print('need to check this and set correct values')
        if (current_time - self.last_update_leap1 >= self.time_passed):
            self.hand_left_leap1=[]
            self.hand_right_leap1=[]
            print('leap 1 value value too old')
        if (current_time - self.last_update_leap2 >= self.time_passed):
            self.hand_left_leap2=[]
            self.hand_right_leap2=[]
            print('leap 2 value value too old')
        print('check passed')'''
        fused_left_hand = fuse_hand(self.hand_left_leap1, self.hand_left_leap2)
        fused_right_hand = fuse_hand(self.hand_right_leap1, self.hand_right_leap2)

        self.frame_counter += 1
        msg = self.create_frame(fused_left_hand, fused_right_hand, current_time)

        self.publisher_.publish(msg)
        print('Published fused data')



def main(args=None):
    rclpy.init(args=args)
    leap_fusion = LeapFusion()
    rclpy.spin(leap_fusion)
    leap_fusion.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
