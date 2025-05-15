import rclpy
import rclpy.logging
import numpy as np
import threading
import time

import matplotlib.pyplot as plt

from jaka_interface.interface import JakaInterface
from jaka_interface.data_types import MoveMode
from jaka_messages.msg import LeapHand
from jaka_safe_control.predictive_safe_controller import PredictiveSafeController
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
        self.hand_pos = np.array([-0.4, 0.0, 0.3])
        self.hand_radius = 0.05

        # Controller
        self.controller = PredictiveSafeController(self.jaka_interface.robot, logger=self.logger)
                
        self.home = np.array([-400, 300, 300, -np.pi, 0, -20*np.pi/180])
        self.robot.disable_servo_mode()
        #self.robot.linear_move(self.home, MoveMode.ABSOLUTE, 100, True)
        self.robot.enable_servo_mode()

        self.h_history = []
        self.t_history = []

        self.q_target = self.robot.get_joint_position()

    def control_loop(self):   
        start = time.time()

        qd_out = self.controller.update_w_cbf(self.t, self.hand_pos)
        if max(abs(qd_out * self.dt)) > 1e-1:
            raise RuntimeError(f"{qd_out * self.dt}")

        q = self.robot.get_joint_position()

        h_val = self.controller.h_func(self.t, q)

        # 2) store it
        self.h_history.append(h_val)
        self.t_history.append(self.t)

        self.q_target += qd_out * self.dt

        self.jaka_interface.robot.servo_j(self.q_target, MoveMode.ABSOLUTE)

        actual_dt = time.time() - start
        self.t += actual_dt#self.dt
    
    def leap_hand_callback(self, msg: LeapHand):
        self.hand_pos = np.array([msg.x, msg.y, msg.z])
        self.hand_radius = msg.radius        

    def loginfo(self, msg):
        self.logger.info(str(msg))
    

    def plot_and_exit(self):
        # Grab your history
        h = np.array(self.h_history)
        t = np.array(self.t_history)

        # 1) Make a big figure
        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)    

        # 3) Plot with thick lines and big markers
        ax.plot(t, h,
                linewidth=3,
                marker='o',
                markersize=6,
                label='h(t)')
        ax.axhline(0,
                linestyle='--',
                linewidth=2,
                label='h = 0')

        # 4) Huge fonts on labels, title, ticks
        ax.set_xlabel('Time [s]', fontsize=18)
        ax.set_ylabel('Safety margin h', fontsize=18)
        ax.set_title('CBF Safety Margin over Time', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(True)

        plt.tight_layout()
        plt.show(block=True)

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
    
    try:
        spin_node(node)
    except KeyboardInterrupt:
        # User pressed Ctrl-C
        node.get_logger().info('Shutdown requested, plotting h-values...')
        node.plot_and_exit()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if node.publish_robot_state:
        interface_thread.join()
        interface.destroy_node()
    node.destroy_node()

    rclpy.shutdown()

if __name__=='__main__':
    main()