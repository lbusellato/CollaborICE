import rclpy
import rclpy.logging
import numpy as np
import cvxpy as cp
import json
import threading
import time

from std_msgs.msg import String
from jaka_interface.interface import JakaInterface
from jaka_interface.data_types import MoveMode
from jaka_messages.msg import LeapHand
from jaka_safe_control.path_functions import triangle_wave 
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor

class JAKA(Node):
    def __init__(self):
        # Init node
        super().__init__('triangle_Wave_node')

        self.declare_parameter('publish_robot_state', rclpy.Parameter.Type.BOOL)
        self.publish_robot_state = self.get_parameter('publish_robot_state').value
        
        # Init the robot interface
        self.jaka_interface = JakaInterface(publish_state=self.publish_robot_state)
        self.jaka_interface.initialize()

        # Logging
        self.logger = rclpy.logging.get_logger('triangle_Wave_node')

        self.t = 0
        self.dt = 0.008
        self.control_loop_timer = self.create_timer(self.dt, self.control_loop) 

        self.min_hand_confidence = 0.33
        self.hand_radius = 0
        self.max_joint_velocity = 100 #0.95 * np.pi
        self.qd_prev = np.zeros(6)

        # Obstacle position
        self.obstacle_pos_sub = self.create_subscription(LeapHand, '/jaka/control/hand', self.leap_hand_callback, 1)
        self.obstacle_pos = None
        
        # Go to the starting position
        self.home = np.array([-212, -467, 300, -np.pi, 0, -20*np.pi/180])
        self.jaka_interface.robot.disable_servo_mode()
        self.jaka_interface.robot.linear_move(self.home, MoveMode.ABSOLUTE, 200, True)
        self.jaka_interface.robot.enable_servo_mode()

        self.path_function = triangle_wave

    def control_loop(self):
        q = np.array(self.jaka_interface.robot.joint_position)
        tcp = np.array(self.jaka_interface.robot.tcp_position)
            
        q_des = np.array(self.path_function(self.t))

        qd_des = (q_des - q) / self.dt
        
        if self.obstacle_pos is not None:
            qd_safe = self.safe_controller(qd_des, tcp[:3], self.obstacle_pos)
            q_out = q + qd_safe * self.dt
        else:
            q_out = q_des

        self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)

        self.t += self.dt
    
    #########################################
    #                                       #
    # Control Barrier Functions             #
    #                                       #
    #########################################

    def safe_controller(self, qd_des, tcp, obstacle, min_distance: int=100):

        qd_opt = cp.Variable(len(qd_des))

        # Define the cost function (same as MATLAB)
        H = np.eye(len(qd_des))  # Identity matrix
        f = -qd_des  # Linear term

        # Quadratic cost function
        objective = cp.Minimize((1/2) * cp.quad_form(qd_opt, H) + f.T @ qd_opt)

        J = self.jaka_interface.robot.jacobian()

        d = np.linalg.norm(tcp - obstacle)
                
        gamma = 0.8  
        h = d - self.hand_radius - min_distance
        n = (tcp - obstacle) / d  # Normal vector
        grad_h = n @ J[:3, :]  # Compute gradient
        constraints = [
            -grad_h.reshape(1, -1) @ qd_opt <= gamma * h,
            qd_opt >= -self.max_joint_velocity,
            qd_opt <= self.max_joint_velocity]

        # Solve the QP problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        # Get the optimized joint velocity
        if qd_opt.value is None:
            raise RuntimeError('solve QP failed')
        qd_optimal = qd_opt.value

        alpha = 0.2  # Smoothing factor (0 < alpha <= 1), adjust as needed
        qd_smoothed = alpha * qd_optimal + (1 - alpha) * self.qd_prev
        self.qd_prev = qd_smoothed 

        return qd_smoothed
    
    #########################################
    #                                       #
    # Subscriber callbacks                  #
    #                                       #
    #########################################
    
    def leap_hand_callback(self, msg: LeapHand):
        self.hand_pos = np.array([msg.x, msg.y, msg.z])
        self.hand_radius = msg.radius        

    #########################################
    #                                       #
    # Utils                                 #
    #                                       #
    #########################################

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