import rclpy
import rclpy.logging
import numpy as np
import threading

from jaka_interface.interface import JakaInterface
from jaka_interface.data_types import MoveMode
from jaka_messages.msg import LeapHand
from jaka_safe_control.vanilla_safe_controller import VanillaSafeController
from jaka_safe_control.path_functions import triangle_wave 
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

class JAKA(Node):
    def __init__(self):
        # Init node
        super().__init__('triangle_wave_node')

        self.declare_parameter('publish_robot_state', rclpy.Parameter.Type.BOOL)
        self.publish_robot_state = self.get_parameter('publish_robot_state').value
        
        # Init the robot interface
        self.jaka_interface = JakaInterface(publish_state=self.publish_robot_state)
        self.jaka_interface.initialize()

        # Logging
        self.logger = rclpy.logging.get_logger('triangle_wave_node')

        self.t = 0
        self.dt = 0.008
        self.control_loop_timer = self.create_timer(self.dt, self.control_loop) 

        # CBF parameters
        gamma = 0.8
        min_hand_distance = 100
        max_joint_velocity = 100 

        # Parameter for the output velocity low-pass filter
        qd_smoothing_factor = 0.3

        # Hand position
        self.hand_pos_sub = self.create_subscription(LeapHand, '/jaka/control/hand', self.leap_hand_callback, 1)
        self.hand_pos = None
        self.hand_radius = 0

        # Controller
        self.controller = VanillaSafeController(self.jaka_interface.robot, 
                                                gamma,
                                                min_hand_distance,
                                                max_joint_velocity,
                                                qd_smoothing_factor,
                                                self.dt)
        
        # Go to the starting position, after waiting for user confirmation
        input('Press any key to start. The robot will move to the home position!')
        self.home = np.array([-400, 300, 300, -np.pi, 0, -20*np.pi/180])
        self.jaka_interface.robot.disable_servo_mode()
        self.jaka_interface.robot.linear_move(self.home, MoveMode.ABSOLUTE, 200, True)
        self.jaka_interface.robot.enable_servo_mode()

        self.path_function = triangle_wave

    def control_loop(self):            
        tcp_des = np.array(self.path_function(self.t))
        q_des = self.jaka_interface.robot.kine_inverse(None, tcp_des)

        q_out = self.controller.update(q_des, self.hand_pos, self.hand_radius)

        self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)

        self.t += self.dt
    
    def leap_hand_callback(self, msg: LeapHand):
        self.hand_pos = np.array([msg.x, msg.y, msg.z])
        self.hand_radius = msg.radius        

    def loginfo(self, msg):
        self.logger.info(str(msg))

def spin_node(node):
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.spin()

def main():
    rclpy.init()
    node = JAKA()
    if node.publish_robot_state:
        # If we are visualizing, spin the interface node in a separate thread
        interface = node.jaka_interface
        interface_thread = threading.Thread(target=spin_node, args=[interface, ], daemon=True)
        interface_thread.start()
    
    spin_node(node)

    if node.publish_robot_state:
        interface_thread.join()
        interface.destroy_node()
    node.destroy_node()

    rclpy.shutdown()

if __name__=='__main__':
    main()