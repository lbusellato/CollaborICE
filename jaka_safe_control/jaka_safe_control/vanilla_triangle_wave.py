import rclpy
import rclpy.logging
import numpy as np
import cvxpy as cp
import threading

from jaka_interface.interface import JakaInterface
from jaka_interface.data_types import MoveMode
from jaka_messages.msg import LeapHand
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
        self.gamma = 0.8
        self.min_hand_distance = 100
        self.max_joint_velocity = 100 

        # Parameters for the output velocity low-pass filter
        self.qd_prev = np.zeros(6)
        self.qd_smoothing_factor = 0.3

        # Hand position
        self.hand_pos_sub = self.create_subscription(LeapHand, '/jaka/control/hand', self.leap_hand_callback, 1)
        self.hand_pos = None
        self.hand_radius = 0
        
        # Go to the starting position
        self.home = np.array([-400, 300, 300, -np.pi, 0, -20*np.pi/180])
        self.jaka_interface.robot.disable_servo_mode()
        self.jaka_interface.robot.linear_move(self.home, MoveMode.ABSOLUTE, 200, True)
        self.jaka_interface.robot.enable_servo_mode()

        self.path_function = triangle_wave

    def control_loop(self):
        q = np.array(self.jaka_interface.robot.joint_position)
        tcp = np.array(self.jaka_interface.robot.tcp_position)
            
        tcp_des = np.array(self.path_function(self.t))
        q_des = self.jaka_interface.robot.kine_inverse(None, tcp_des)

        qd = (q_des - q) / self.dt

        if self.hand_pos is not None:
            qd = self.safe_controller(qd, tcp[:3])

        qd = self.qd_smoothing_factor * qd + (1 - self.qd_smoothing_factor) * self.qd_prev
        self.qd_prev = qd 

        q_out = q + qd * self.dt

        self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)

        self.t += self.dt
    
    #########################################
    #                                       #
    # Control Barrier Functions             #
    #                                       #
    #########################################

    def safe_controller(self, qd_des, tcp):

        qd_opt = cp.Variable(len(qd_des))

        # Define the cost function (same as MATLAB)
        H = np.eye(len(qd_des))  # Identity matrix
        f = -qd_des  # Linear term

        # Quadratic cost function
        objective = cp.Minimize((1/2) * cp.quad_form(qd_opt, H) + f.T @ qd_opt)

        J = self.jaka_interface.robot.jacobian()

        d = np.linalg.norm(tcp - self.hand_pos)
                
        h = d - self.hand_radius - self.min_hand_distance
        n = (tcp - self.hand_pos) / d  # Normal vector
        grad_h = n @ J[:3, :]  # Compute gradient
        constraints = [
            -grad_h.reshape(1, -1) @ qd_opt <= self.gamma * h,
            qd_opt >= -self.max_joint_velocity,
            qd_opt <= self.max_joint_velocity]

        # Solve the QP problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        # Get the optimized joint velocity
        if qd_opt.value is None:
            raise RuntimeError('solve QP failed')

        return qd_opt.value
    
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