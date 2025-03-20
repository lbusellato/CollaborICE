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
        super().__init__('triangle_Wave_node')

        self.declare_parameter('publish_robot_state', rclpy.Parameter.Type.BOOL)
        self.publish_robot_state = self.get_parameter('publish_robot_state').value
        
        # Init the robot interface
        self.jaka_interface = JakaInterface(publish_state=self.publish_robot_state)
        self.jaka_interface.initialize()

        # Logging
        self.logger = rclpy.logging.get_logger('triangle_Wave_node')

        self.tau = 0
        self.dt = 0.008
        self.control_loop_timer = self.create_timer(self.dt, self.control_loop) 


        # Hand position
        self.hand_pos_sub = self.create_subscription(LeapHand, '/jaka/control/hand', self.leap_hand_callback, 1)
        self.hand_pos = [-400, 0, 300]
        self.hand_radius = 80
        self.hand_forecast = []

        # CBF parameters
        self.T = 1 # seconds in the future
        self.min_hand_distance = 100
        self.gamma = 0.8
        self.gamma_margin = 1.0
        self.max_joint_velocity = 10

        # Parameters for the output velocity low-pass filter
        self.qd_prev = np.zeros(6)
        self.qd_smoothing_factor = 0.3
        
        # Go to the starting position
        self.home = np.array([-400, 300, 300, -np.pi, 0, -20*np.pi/180])
        self.jaka_interface.robot.disable_servo_mode()
        #self.jaka_interface.robot.linear_move(self.home, MoveMode.ABSOLUTE, 200, True)
        self.jaka_interface.robot.enable_servo_mode()

        self.path_function = triangle_wave

    def control_loop(self):
        q = np.array(self.jaka_interface.robot.joint_position)
        tcp = np.array(self.jaka_interface.robot.tcp_position)
            
        tcp_des = np.array(self.path_function(self.tau))
        q_des = self.jaka_interface.robot.kine_inverse(None, tcp_des)

        qd = (q_des - q) / self.dt
        
        if self.hand_pos is not None:
            qdd = self.safe_controller(qd, tcp[:3])
            if qdd is not None: qd = qdd

        qd = self.qd_smoothing_factor * qd + (1 - self.qd_smoothing_factor) * self.qd_prev
        self.qd_prev = qd 

        q_out = q + qd * self.dt

        #self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)

        self.tau += self.dt
    
    #########################################
    #                                       #
    # Control Barrier Functions             #
    #                                       #
    #########################################

    def safe_controller(self, qd_des, tcp):    
        # Compute predictive CBF (H*) based on current state (using tcp as x)
        H_star_val = self.Hstar(self.tau, tcp)
        
        # Compute approximate gradient of H*.
        # We approximate ∇H* by taking the gradient of h at tau0.
        predicted_tcp = self.path_function(self.tau, 0, tcp)[:3]
        predicted_hand = self.predict_hand(self.tau, 0, tcp)
        d = np.linalg.norm(predicted_tcp - predicted_hand)
        if d > 0:
            n = (predicted_tcp - predicted_hand) / d  # Unit vector
        else:
            n = np.zeros_like(predicted_tcp)
        
        # Get the robot Jacobian (assumed to relate joint velocities to tcp velocities)
        J = self.jaka_interface.robot.jacobian()
        # Approximate gradient of H* with respect to joint velocities:
        grad_H = n @ J[:3, :]  # Row vector
        
        # Set up the QP:
        # Decision variable: joint velocity command
        qd_opt = cp.Variable(len(qd_des))
        
        # Objective: stay as close as possible to the desired qd_des
        H = np.eye(len(qd_des))
        f = -qd_des
        objective = cp.Minimize((1/2) * cp.quad_form(qd_opt, H) + f.T @ qd_opt)
        
        # CBF constraint: enforce -grad_H * qd <= gamma_margin * H*
        constraints = [
            -grad_H @ qd_opt <= self.gamma_margin * H_star_val,
            qd_opt >= -self.max_joint_velocity,
            qd_opt <= self.max_joint_velocity
        ]
        
        # Solve the QP
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        
        if qd_opt.value is None:
            pass #raise RuntimeError("QP failed in predictive CBF safe controller")

        return qd_opt.value

    def H(self, t: float, x: list)->list:
        M = self.find_maximizers(t, x)
        return [self.hp(m[0], t, x) for m in M]

    def Hstar(self, t: float, x: list):
        M = self.find_maximizers(t, x)
        Mstar = M[0]
        return self.hp(Mstar[0], t, x)

    def hp(self, tau: float, t: float, x: list)->float:
        def m(val):
            # Margin function
            return self.gamma_margin * val
        return self.h(tau, t, x) - m(self.R(tau, t, x) - t)

    def R(self, tau: float, t: float, x: list)->float:
        if self.h(tau, t, x) > 0:
            return self.R1(tau, t, x)
        else:
            return tau
    
    def R1(self, tau: float, t: float, x: list)->float:
        horizon = np.arange(tau, tau+self.T, self.dt)
        h_values = [self.h(_tau, t, x) for _tau in horizon]
        
        if h_values[0] > 0:
            return t # Already unsafe at the start
        
        # Search for the first interval where h goes from negative to positive
        for i in range(len(h_values) - 1):
            if h_values[i] < 0 and h_values[i+1] > 0:
                # Perform linear interpolation to estimate the zero crossing:
                t0, t1 = horizon[i], horizon[i+1]
                h0, h1 = h_values[i], h_values[i+1]
                # alpha is the fraction from t0 at which h=0 occurs:
                alpha = -h0 / (h1 - h0)
                R1 = t0 + alpha * (t1 - t0)
                return R1

        # If h remains safe throughout the horizon, return tau as the root.
        return tau

    def find_maximizers(self, t: float, x: list)->list:
        # Compute the safety function over the discretized time horizon
        horizon = np.arange(t, t+self.T, self.dt)
        h_values = [self.h(tau, t, x) for tau in horizon]
        
        # Compute the maximizers of h 
        M = []
        n = len(h_values)
        if h_values[0] > h_values[1]: M.append((0, h_values[0]))
        for i in range(1, n - 1):
            if h_values[i - 1] < h_values[i] > h_values[i + 1] or h_values[i - 1] < h_values[i] == h_values[i + 1]:
                M.append((i * self.dt, h_values[i]))
        if h_values[-1] > h_values[-2]: M.append(((n - 1) * self.dt, h_values[n - 1]))
        return M
    
    def h(self, tau: float, t: float, x: list)->float:
        predicted_tcp = self.path_function(tau, t, x)[:3] * 1000
        predicted_hand = self.predict_hand(tau, t, x)
        if predicted_hand is not None:
            d = np.linalg.norm(predicted_tcp - predicted_hand)
            h = (self.hand_radius + self.min_hand_distance) - d
        else:
            h = -500 # Random negative variable to signal we are in a safe condition (no hand to avoid)
        return h

    def predict_hand(self, tau: float, t: float, x: list):
        if len(self.hand_forecast) > 0:
            horizon = np.linspace(t, t+self.T, len(self.hand_forecast))
            idx = (np.abs(horizon - tau)).argmin()
            return self.hand_forecast[idx]
        else:
            return self.hand_pos # Fallback to the current position if no forecasting is available
    
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