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

        self.max_joint_velocity = 100 
        self.qd_prev = np.zeros(6)

        # Hand position
        self.hand_pos_sub = self.create_subscription(LeapHand, '/jaka/control/hand', self.leap_hand_callback, 1)
        self.hand_pos = None
        self.hand_radius = 0
        self.hand_forecast = []

        # CBF parameters
        self.T = 1 # seconds in the future
        self.min_distance = 100
        self.gamma_margin = 1.0
        
        # Go to the starting position
        self.home = np.array([-212, -467, 300, -np.pi, 0, -20*np.pi/180])
        self.jaka_interface.robot.disable_servo_mode()
        self.jaka_interface.robot.linear_move(self.home, MoveMode.ABSOLUTE, 200, True)
        self.jaka_interface.robot.enable_servo_mode()

        self.path_function = triangle_wave

    def control_loop(self):
        q = np.array(self.jaka_interface.robot.joint_position)
        tcp = np.array(self.jaka_interface.robot.tcp_position)
            
        q_des = np.array(self.path_function(self.tau))

        qd = (q_des - q) / self.dt
        
        if self.hand_pos is not None:
            qd = self.safe_controller(qd, tcp[:3], self.hand_pos)
            # HACK to cover the case in which the hand disappears or the forecasting stops working
            self.hand_pos = None
            self.hand_forecast = []

        alpha = 0.2  # Smoothing factor (0 < alpha <= 1), adjust as needed
        qd_smoothed = alpha * qd + (1 - alpha) * self.qd_prev
        self.qd_prev = qd_smoothed 

        q_out = q + qd * self.dt

        self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)

        self.tau += self.dt
    
    #########################################
    #                                       #
    # Control Barrier Functions             #
    #                                       #
    #########################################

    def safe_controller(self, qd_des, tcp):
        # Current time from our controller
        t_current = self.tau
        
        # Compute predictive CBF (H*) based on current state (using tcp as x)
        H_star_val = self.Hstar(self.t, tcp)
        
        # Compute approximate gradient of H*.
        # We approximate ∇H* by taking the gradient of h at tau0.
        predicted_tcp = self.path_function(t_current, 0, tcp)
        predicted_hand = self.predict_hand(t_current, 0, tcp)
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
        cost = 0.5 * cp.quad_form(qd_opt, np.eye(len(qd_des))) - qd_des.T @ qd_opt
        objective = cp.Minimize(cost)
        
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
            raise RuntimeError("QP failed in predictive CBF safe controller")
        
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
        horizon = np.arange(t, tau, self.dt)
        h_values = [self.h(_tau, t) for _tau in horizon]
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
        h_values = [self.h(tau, t) for tau in horizon]
        # Compute the maximizers of h
        M = [(i * self.dt, h_val) for i, h_val in enumerate(h_values[1:-1], 1) 
             if h_values[i - 1] <= h_val <= h_values[i + 1] ]
        # Keep only the maximizers of h (only the first one if a plateau is found)
        Mstar = M[0] # Always keep the first time an unsafe situation is detected
        M = [m for i, m in enumerate(M[1:], 1) if m[1] != M[i - 1][1]]
        M.insert(0, Mstar)
        return M
    
    def h(self, tau: float, t: float, x: list)->float:
        predicted_tcp = self.path_function(tau, t, x)
        predicted_hand = self.hand_forecast(tau, t, x)
        if predicted_hand is not None:
            d = np.linalg.norm(predicted_tcp - predicted_hand)
            h = d - (self.hand_radius + self.min_distance)
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