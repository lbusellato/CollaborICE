import rclpy
import rclpy.logging
import numpy as np
import threading

from jaka_interface.interface import JakaInterface
from jaka_interface.data_types import MoveMode
from jaka_messages.msg import LeapHand
from jaka_safe_control.predictive_safe_controller import PredictiveSafeController
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
        self.robot = self.jaka_interface.robot

        # Logging
        self.logger = rclpy.logging.get_logger('triangle_wave_node')

        self.t = 0
        self.dt = 0.008
        #self.control_loop_timer = self.create_timer(self.dt, self.control_loop) 

        # Hand position
        self.hand_pos_sub = self.create_subscription(LeapHand, '/jaka/control/hand', self.leap_hand_callback, 1)
        self.hand_pos = None
        self.hand_radius = 0

        # Controller
        self.controller = PredictiveSafeController(self.jaka_interface.robot)
                
        self.home = np.array([-400, 300, 300, -np.pi, 0, -20*np.pi/180])
        self.robot.disable_servo_mode()
        #self.robot.linear_move(self.home, MoveMode.ABSOLUTE, 100, True)
        self.robot.enable_servo_mode()


    def control_loop(self):           
        qd_out = self.controller.update(self.t, self.hand_pos)

        if max(abs(qd_out * self.dt)) > 1e-1:
            raise RuntimeError

        q = self.robot.joint_position

        q_out = q + qd_out * self.dt
        
        #self.loginfo(qd_out)
        #self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)

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