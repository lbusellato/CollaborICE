import rclpy
import rclpy.logging
import numpy as np
import threading
import cvxpy as cp

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
        self.gamma = 0.8
        self.min_hand_distance = 100
        self.max_joint_velocity = 100 
        self.h_func_delta = 0.01
        self.T = 1

        # Parameter for the output velocity low-pass filter
        self.qd_smoothing_factor = 0.3
        self.qd_prev = np.zeros(6)

        # Hand position
        self.hand_pos_sub = self.create_subscription(LeapHand, '/jaka/control/hand', self.leap_hand_callback, 1)
        self.hand_pos = None
        self.hand_radius = 0
        
        # Go to the starting position, after waiting for user confirmation
        input('Press any key to start. The robot will move to the home position!')
        self.home = np.array([-400, 300, 300, -np.pi, 0, -20*np.pi/180])
        self.jaka_interface.robot.disable_servo_mode()
        self.jaka_interface.robot.linear_move(self.home, MoveMode.ABSOLUTE, 200, True)
        self.jaka_interface.robot.enable_servo_mode()

        self.path_function = triangle_wave

    def control_loop(self):            
        q = np.array(self.robot.joint_position)
        tcp = np.array(self.robot.tcp_position)[:3]

        tcp_des = np.array(self.path_function(self.t))
        q_des = self.jaka_interface.robot.kine_inverse(None, tcp_des)

        qd = (q_des - q) / self.dt

        if self.hand_pos is not None:
            x = np.vstack([tcp, self.hand_pos])
            qd = self.calculate_U(self.t, x, qd)

        qd = self.qd_smoothing_factor * qd + (1 - self.qd_smoothing_factor) * self.qd_prev
        self.qd_prev = qd

        q_out = q + qd * self.dt

        self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)

        self.t += self.dt

    def calculate_U(self, t, x, mu):
        H, dHdt, dHdu = self.Hstar_func(t, x)

        J = np.eye(6)
        F = -mu

        A = dHdu
        b = -self.gamma * H - dHdt
    
        du = cp.Variable(6)

        objective = cp.Minimize(0.5 * cp.quad_form(du, J) + F.T @ du)
        constraints = [A @ du <= b,
            du >= -self.max_joint_vel,
            du <= self.max_joint_vel]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if du.value is None or np.any(np.isnan(du.value)):
            raise RuntimeError('QP solve failed')
        
        return du

    def Hstar_func(self, t, x):
        pass

    def find_max(self, t, x):
        tau, neg_h_of_tau = self.my_fmincon(
            func=lambda z: -self.h_func(z, self.path_func(z, t, x)),
            t1=t,
            t2=t + self.T
        )
        h_of_tau = -neg_h_of_tau
        
        delta = 0.01
        dx = np.zeros(len(x))
        for i in range(len(x)):
            x_new = x.copy()
            x_new[i] += delta
            # Re-run the search in a smaller window [tau-0.2, tau+0.2], as in MATLAB
            new_tau, _ = self.my_fmincon(
                func=lambda z: -self.h_func(z, self.path_func(z, t, x_new)),
                t1=tau - 0.01,
                t2=tau + 0.01
            )
            dx[i] = (new_tau - tau) / delta

        return tau, h_of_tau, dx
    
    def path_func(self, tau, t, x):
        x_tcp = np.array(self.path_function(tau, t))
        x_obs = self.hand_pos # TODO plug in the forecasting at time t + tau here

        p = np.vstack([x_tcp, x_obs])

        

    def my_fmincon(self, func, t1, t2):
        N = 10
        tol = 1e-5
        
        # We'll iteratively shrink [t1, t2] based on the grid search
        while (t2 - t1) > tol:
            # Create a grid of N points from t1 to t2
            t_grid = np.linspace(t1, t2, N)
            # Evaluate func at each point
            fvals = np.array([func(z) for z in t_grid])
            # Find the index of the smallest function value
            idx_min = np.argmin(fvals)
            
            # Narrow the interval around the minimum
            if idx_min == 0:
                # If the best is at the left boundary, shift interval slightly right
                t1, t2 = t_grid[0], t_grid[1]
            elif idx_min == N - 1:
                # If the best is at the right boundary, shift interval slightly left
                t1, t2 = t_grid[N - 2], t_grid[N - 1]
            else:
                # Otherwise, zoom in around the minimum
                t1, t2 = t_grid[idx_min - 1], t_grid[idx_min + 1]

        # After the interval is small enough, pick the midpoint (or the best in the last grid)
        # For consistency with MATLAB, we do one final evaluation:
        t_grid = np.linspace(t1, t2, N)
        fvals = np.array([func(z) for z in t_grid])
        idx_min = np.argmin(fvals)
        xval = t_grid[idx_min]
        fval = fvals[idx_min]
        return xval, fval

    def h_func(self, t, x):
        out = self.h_func_real(x)

        dx = []
        for i in range(2):
            x_new1 = x[:, i]
            x_new1 += self.h_func_delta
            x_new2 = x[:, i]
            x_new2 -= self.h_func_delta
            dx.append((self.h_func_real(x_new1) + self.h_func_real(x_new1))/(2*self.h_func_delta))
        dx = np.array(dx)

        return out, dx

    def h_func_real(self, x):
        distance = np.linalg.norm(x[:,0] - x[:,1])
        return (self.hand_radius + self.min_hand_distance) - distance
    
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