import rclpy
import rclpy.logging
import numpy as np
import threading
import time
import datetime
import csv
from os.path import dirname, join
from ament_index_python.packages import get_package_share_directory
from scipy.interpolate import interp1d
import cProfile
import pstats

import matplotlib.pyplot as plt

from jaka_interface.interface import JakaInterface
from jaka_interface.pose_conversions import rpy_to_rot_matrix, leap_to_jaka
from jaka_interface.data_types import MoveMode
from jaka_messages.msg import LeapHand
from jaka_safe_control.predictive_safe_controller import PredictiveSafeController
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from std_msgs.msg import String
import json
from scipy.optimize import minimize_scalar
import cvxpy as cp

MOVING = True

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
        self.dt = 0.05

        # Hand position
        self.hand_pos_sub = self.create_subscription(LeapHand, '/jaka/control/hand', self.leap_hand_callback, 1)
        self.hand_forecast_sub = self.create_subscription(String, '/applications/hand_forecasting/kalman', self.forecast_callback, 1)
        self.current_hand_pos = np.zeros(3)
        self.current_hand_radius = 0.0

        # PCBF parameters
        self.T = 1  # Prediction horizon
        self.min_safety_distance = 0.025
        self.h_max_estimate = self.current_hand_radius + self.min_safety_distance # Derived from h_func definition
        self.m = self.h_max_estimate / self.T**2 # Barrier function parameter
        self.perturb_delta = 1e-4  # For numerical gradient computation
        self.alpha = 20       
        self.lamda = 10 
        self.max_joint_vel = np.pi / 4

        self.smoothing = 0.2   
        self.prev_u = np.zeros(6)

        self.n_joints = 6
        self.state_len = 6

        self.targetA = np.array([-0.4, 0.0, 0.3, -np.pi, 0, -20*np.pi/180])
        self.targetB = np.array([-0.4, 0.5, 0.3, -np.pi, 0, -20*np.pi/180])
        self.tcp_target = self.targetA
        self.pos_tolerance = 1e-3
        self.rot_tolerance = 1e-2
        self.pos_gain = 1  # position gain
        self.rot_gain = 1  # orientation gain
                
        self.home = np.array([-400, 500, 300, -np.pi, 0, -20*np.pi/180])

        self.rate = self.declare_parameter('rate', 100.0).value

        if MOVING:
            self.control_loop_timer = self.create_timer(1.0/self.rate, self.control_loop) 
            self.robot.disable_servo_mode()
            self.robot.linear_move(self.home, MoveMode.ABSOLUTE, 100, True)
            self.robot.enable_servo_mode()

        time.sleep(1)

        self.workspace_limits = {
            'x_min': -0.5, 'x_max': 0.1, # don't hit the operator/camera rig
            'y_min': -0.5, 'y_max': 0.8, # don't hit the milling
            'z_min':  0.3           # don't go below table height
        }

        self.q_target = self.robot.get_joint_position()

        self.hand_forecast = []

        # Data recording lists
        self.time_data = []
        self.ee_pos_data = []            # TCP x,y,z
        self.u_nom_data = []             # nominal joint velocities
        self.u_data = []                 # safe (CBF/PCBF) joint velocities
        self.h_data = []                 # safety function values
        self.hstar_data = []             # predictive CBF values (if using PCBf)

        self.last_time = self.get_clock().now()
        self.iter = 0
        self.dt_history = np.zeros(10)

        self.converged = False

    def control_loop(self):   
        
        self.t += self.dt

        current_state = self.robot.get_joint_position()

        if not self.converged:
            u_nom = self.mu_func(current_state, 0.008)

            #u, h_star = self.calculate_u_pcbf(self.t, current_state)
            u, h = self.calculate_u_cbf(self.t, current_state)
            h_star = None

            if max(abs(u * self.dt)) > 5e-1:
                raise RuntimeError(f"{u * self.dt}")
            
            u_out = self.smoothing * self.prev_u + (1 - self.smoothing) * u
            self.prev_u = u

            self.q_target += u_out * 0.008#self.dt
            h = self.h_func(self.t, self.q_target)

            #now = self.get_clock().now()
            #self.dt_history[self.iter] = (now - self.last_time).nanoseconds * 1e-9
            #self.iter += 1
            #if self.iter == 10:
            #    self.loginfo(f"Loop frequency: {1 / np.mean(self.dt_history)}")
            #    self.iter = 0
            #self.last_time = now

            self.jaka_interface.robot.servo_j(self.q_target, MoveMode.ABSOLUTE)

            tcp_pose = self.robot.kine_forward(current_state)

            pos_error = np.linalg.norm(tcp_pose.t - self.tcp_target[:3])
            current_rpy = np.array(tcp_pose.rpy())
            rpy_error = np.array([self.angle_diff(current_rpy[i], self.tcp_target[3+i]) for i in range(3)])
            rot_error = np.linalg.norm(rpy_error)
            
            self.converged = (pos_error < self.pos_tolerance and rot_error < self.rot_tolerance)

            self.time_data.append(self.t)
            self.ee_pos_data.append(tcp_pose.t.copy())
            self.u_nom_data.append(u_nom.copy())
            self.u_data.append(u.copy())
            self.h_data.append(h)
            if h_star is not None:
                self.hstar_data.append(h_star)

        else:
            if np.allclose(self.tcp_target, self.targetA, rtol=1e-2):
                self.tcp_target = self.targetB
            else:
                self.tcp_target = self.targetA  
            self.converged = False

    def save_data(self):
        """
        Save recorded data to a timestamped CSV file using the standard csv module.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.csv"
        package_share_path = get_package_share_directory('jaka_safe_control')

        filepath = join(package_share_path, 'logs', filename)

        # Prepare header and rows
        ee = np.array(self.ee_pos_data)
        u_nom = np.array(self.u_nom_data)
        u_safe = np.array(self.u_data)
        headers = ['time', 'ee_x', 'ee_y', 'ee_z']
        headers += [f'u_nom_{j}' for j in range(u_nom.shape[1])]
        headers += [f'u_safe_{j}' for j in range(u_safe.shape[1])]
        headers.append('h')
        if len(self.hstar_data) == len(self.time_data):
            headers.append('hstar')

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            for idx, t in enumerate(self.time_data):
                row = [t]
                row += list(ee[idx, :])
                row += list(u_nom[idx, :])
                row += list(u_safe[idx, :])
                row.append(self.h_data[idx])
                if len(self.hstar_data) == len(self.time_data):
                    row.append(self.hstar_data[idx])
                writer.writerow(row)

        self.loginfo(f"Data saved to {filepath}")

    def plot_data(self):
        # Convert to arrays
        t = np.array(self.time_data)
        ee = np.array(self.ee_pos_data)  # shape (N,3)
        u_nom = np.array(self.u_nom_data)  # shape (N,6)
        u_safe = np.array(self.u_data)     # shape (N,6)
        h = np.array(self.h_data)
        has_hstar = len(self.hstar_data) == len(self.time_data)
        if has_hstar:
            hstar = np.array(self.hstar_data)

        # 1) End-effector position
        plt.figure()
        plt.plot(t, ee[:,0], label='x')
        plt.plot(t, ee[:,1], label='y')
        plt.plot(t, ee[:,2], label='z')
        plt.title('End-Effector Position vs Time')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.legend()

        # 2) Joint velocities: nominal vs safe
        plt.figure()
        for j in range(u_nom.shape[1]):
            plt.plot(t, u_nom[:,j], linestyle='--', label=f'u_nom[{j}]')
            plt.plot(t, u_safe[:,j], linestyle='-', label=f'u_safe[{j}]')
        plt.title('Joint Velocities: Nominal (-- ) vs Safe (–)')
        plt.xlabel('Time [s]')
        plt.ylabel('Joint velocity [rad/s]')
        plt.legend(loc='upper right', ncol=2)

        # 3) Safety function
        plt.figure()
        plt.plot(t, h, label='h (CBF)')
        if has_hstar:
            plt.plot(t, hstar, label='h* (PCBF)')
        plt.title('Safety Function over Time')
        plt.xlabel('Time [s]')
        plt.ylabel('h value')
        plt.legend()

        plt.show()

    def forecast_callback(self, msg: String):
        msg = json.loads(msg.data)
        forecast = msg.get('future_trajectory')
        forecast = [leap_to_jaka(f)/1000 for f in forecast]
        forecast = [self.current_hand_pos] + forecast
        self.hand_forecast = forecast

    def leap_hand_callback(self, msg: LeapHand):
        self.current_hand_pos = np.array([msg.x, msg.y, msg.z])/1000
        self.current_hand_radius = msg.radius/1000    
        self.h_max_estimate = self.min_safety_distance + self.current_hand_radius
    
    def get_updated_obstacle(self, t):
        # build a simple list: [current_pos, forecast[0], forecast[1], ...]
        u = (t - self.t) / self.T
        
        n = len(self.hand_forecast)

        if n > 0:

            idx = int(np.floor(u * (n - 1)))


            idx = max(0, min(idx, n - 1))

            obstacle_pos = self.hand_forecast[idx]
        else:
            obstacle_pos = self.current_hand_pos

        obstacle_radius = self.current_hand_radius
        
        return obstacle_pos, obstacle_radius

    def h_func(self, t, state):
        """
        Computes the safety margin between a dynamic spherical obstacle and 
        a cylindrical end-effector (EE) volume centered at the TCP.

        The cylinder is aligned with the TCP z-axis, has height h_cyl, and radius r_cyl.
        """
        # EE shape
        r_cyl = 0.045  # cylinder radius [m]
        h_cyl = 0.35   # cylinder height [m]

        # TCP pose and orientation
        tcp_pose = self.robot.kine_forward(state)
        tcp_pos = tcp_pose.t
        tcp_z_axis = tcp_pose.R[:, 2]  # z-axis of EE

        # Endpoint of cylinder axis
        cyl_top = tcp_pos - h_cyl * tcp_z_axis

        # Obstacle position and radius at time t in [tau, tau + T]
        obstacle_pos, obstacle_radius = self.get_updated_obstacle(t)

        # Project obstacle onto cylinder axis
        axis_vec = cyl_top - tcp_pos
        axis_dir = axis_vec / np.linalg.norm(axis_vec)
        vec_to_obs = obstacle_pos - tcp_pos
        proj_length = np.dot(vec_to_obs, axis_dir)
        proj_length_clamped = np.clip(proj_length, 0, np.linalg.norm(axis_vec))
        closest_point_on_axis = tcp_pos + proj_length_clamped * axis_dir

        # Distance from obstacle to closest surface point on the cylinder
        dist_to_cylinder_surface = np.linalg.norm(obstacle_pos - closest_point_on_axis) - r_cyl

        # Safety margin
        return (self.min_safety_distance + obstacle_radius) - dist_to_cylinder_surface
    
    def h_func_with_gradient(self, t, state):
        """Numerically computes the gradient of h with respect to the state q."""
        h_val = self.h_func(t, state)
        dh = np.zeros(self.state_len)
        for i in range(self.state_len):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += self.perturb_delta
            state_minus[i] -= self.perturb_delta
            h_plus = self.h_func(t, state_plus)
            h_minus = self.h_func(t, state_minus)
            dh[i] = (h_plus - h_minus) / (2 * self.perturb_delta)
        return h_val, dh

    def path_func(self, tau, t, state):
        delta_t = tau - t
        u_nom = self.mu_func(state)
        q_pred = state + u_nom * delta_t
        dp_dtau = u_nom      # ∂p/∂τ
        dp_dt    = -u_nom    # ∂p/∂t
        dx       = np.eye(self.state_len)
        return q_pred, dp_dtau, dp_dt, dx
    
    def find_max(self, t, state):
        """
        Finds the time tau within [t, t+T] that maximizes the safety margin h.
        Returns tau, the corresponding h value, and an approximate derivative dtau_dx.
        """
                
        # Solve the optimization using trust-constr with the zero Hessian.
        result = minimize_scalar(
            lambda tau: -self.h_func(tau, self.path_func(tau, t, state)[0]),
            method="bounded",
            bounds=[t, t + self.T],
            options={'xatol': 1e-6}
        )
        tau = result.x
        h_of_tau = -result.fun

        # Approximate dtau/dx via finite differences.
        dtau_dx = np.zeros(self.state_len)
        for i in range(self.state_len):
            state_perturbed = state.copy()
            state_perturbed[i] += self.perturb_delta
            result_perturbed = minimize_scalar(
                lambda tau: -self.h_func(tau, self.path_func(tau, t, state_perturbed)[0]),
                method="bounded",
                bounds=[t, t + self.T],
                options={'xatol': 1e-6}
            )
            tau_perturbed = result_perturbed.x
            dtau_dx[i] = (tau_perturbed - tau) / self.perturb_delta

        return tau, h_of_tau, dtau_dx
    
    def find_zero(self, tau, t, x):
        f = lambda z: self.h_func(z, self.path_func(z, t, x)[0])
        
        eta = self.fzero(f, t, tau)

        p, dp_dtau, _, dp_dx = self.path_func(eta, t, x)
        _, dh = self.h_func_with_gradient(tau, p)
        dx = -dh @ dp_dx / (dh @ dp_dtau)
        
        return eta, dx

    def fzero(self, func, t1, t2, tol=1e-3, rtol=1e-2, max_iter=300):
        """
        use the bisection method to find a zero of *func* in [t1, t2].
        
        Parameters:
            func    (callable): The function for which to find a root.
            t1      (float): The lower bound of the interval.
            t2      (float): The upper bound of the interval.
            tol     (float): The tolerance for the absolute function value.
            max_iter (int): Maximum iterations allowed.
            
        Returns:
            out (float): An approximate zero of func in the interval.
            
        Raises:
            ValueError: If the function does not satisfy the necessary sign conditions.
        """
        h1 = func(t1)
        h2 = func(t2)
        
        if h1 > 0:
            # If the function at the left end is above zero, return t1 as a trivial solution.
            return t1
        elif h2 < 0:
            raise ValueError("Invalid bounds: func(t1) and func(t2) do not bracket a root.")

        # Begin bisection
        tau = (t1 + t2) / 2.0
        h = func(tau)
        count = 0

        while abs(h) > tol:
            count += 1
            if h < 0:
                t1 = tau
            else:
                t2 = tau
            tau = (t1 + t2) / 2.0
            h = func(tau)
            if count > max_iter:
                # Bisection failed, if h satisfies a relaxed constraint, keep it
                if abs(h) < rtol:
                    return tau
                print(f"fzero failed:   h: {h}")
                break

        return tau
    
    def m_func(self, lambda_val):
        self.m = self.h_max_estimate / self.T**2 # Recompute
        m_val = self.m * lambda_val**2
        dm_val = 2 * self.m * lambda_val
        return m_val, dm_val
    
    def hstar_func(self, t, state):
        """
        Computes the predictive CBF value h* and its derivatives for the single integrator (velocity control) case.
        """

        # Find first local maxima of h over the time horizon
        M1star, h_of_M1star, dM1star_dx = self.find_max(t, state)

        # R(tau, t, x)   eqn 9        
        if h_of_M1star <= 0: # First local maxima is safe (eqn 9, case 2)
            R = M1star 
            dR_dx = dM1star_dx
        else: # First local maxima is unsafe, compute root of h before it
            R, dR_dx = self.find_zero(M1star, t, state)
            if np.linalg.norm(dR_dx) >= 10 * np.linalg.norm(dM1star_dx):
                # Switch to M1star when the derivative gets too large
                dR_dx = dM1star_dx

        m, dm = self.m_func(R - t)

        hstar = h_of_M1star - m

        predicted_state, _, _, dp_of_tau_dx = self.path_func(M1star, t, state)
        _, dh_of_tau_dx = self.h_func_with_gradient(M1star, predicted_state)

        g = np.eye(6) # Single integrator model
                
        switching_dt = 1e-4
        if M1star <= t + switching_dt or (M1star <= t + self.T + switching_dt and h_of_M1star <= 0):
            # Case iii - eqn 18
            q_pred_plus = self.path_func(M1star + self.dt, t, state)[0]
            h_plus_delta = self.h_func(M1star + self.dt, q_pred_plus)
            dh_dtau = (h_plus_delta - h_of_M1star) / self.dt
            dtau_dt = 1 # Because the system is of high degree
            dt = dh_dtau * dtau_dt
            du = (dh_of_tau_dx @ dp_of_tau_dx) @ g
        elif M1star < t + self.T:
            # Case i - eqn 16
            dt = dm
            du = (dh_of_tau_dx @ dp_of_tau_dx - dm * dR_dx) @ g
        elif M1star <= t + self.T + switching_dt and h_of_M1star > 0:
            # Case ii - eqn 17
            q_pred_plus = self.path_func(M1star + self.dt, t, state)[0]
            h_plus_delta = self.h_func(M1star + self.dt, q_pred_plus)
            dh_dtau = (h_plus_delta - h_of_M1star) / self.dt
            dtau_dt = 1 # For simplicity
            dt = dm + dh_dtau * dtau_dt
            du = (dh_of_tau_dx @ dp_of_tau_dx - dm * dR_dx) @ g
        else:
            print("Warning: m1star > t+T")
            dt = 0
            du = np.zeros(self.state_len)
        
        return hstar, dt, du
    
    def mu_func(self, state, dt=None):
        """
        Nominal control law for velocity control.
        Uses a task-space proportional controller to generate a joint velocity command.
        """
        q = state
        current_pose = self.robot.kine_forward(q)
        current_pos = current_pose.t
        target_pos = self.tcp_target[:3]
        pos_error = target_pos - current_pos
        pos_error_mag = np.linalg.norm(pos_error)

        rot_current = rpy_to_rot_matrix(current_pose.rpy())
        rot_target = rpy_to_rot_matrix(self.tcp_target[3:6])
        rot_error_mat = rot_target.dot(rot_current.T)
        rot_error = 0.5 * np.array([
            rot_error_mat[2, 1] - rot_error_mat[1, 2],
            rot_error_mat[0, 2] - rot_error_mat[2, 0],
            rot_error_mat[1, 0] - rot_error_mat[0, 1]
        ])
        rot_error_norm = np.linalg.norm(rot_error)
                    
        v_cmd = np.zeros(6)
        
        if pos_error_mag > self.pos_tolerance:
            v_cmd[:3] = self.pos_gain * pos_error
        
        if rot_error_norm > self.rot_tolerance:
            v_cmd[3:] = self.rot_gain * rot_error
            
        J = self.robot.jacobian(q)
        J_pinv = np.linalg.pinv(J)
        q_dot_des = J_pinv @ v_cmd
        
        return q_dot_des
    
    def solve_qp(self, u_nom, q, A, b, P=np.eye(6), F=np.zeros(6)):
        
        u = cp.Variable(self.n_joints)

        objective = cp.Minimize(0.5 * cp.quad_form(u, P) + F @ u)

        J_cart = self.robot.jacobian(q)
        tcp_pos = self.robot.kine_forward(q).t

        ws = self.workspace_limits
        dt = self.dt

        constraints = [
            A @ (u + u_nom) <= b,                                             # CBF constraint
            (u + u_nom) >= -self.max_joint_vel,                               # max velocity constraint
            (u + u_nom) <= self.max_joint_vel,                                # max velocity constraint
            J_cart[0, :] @ (u + u_nom) <= (ws['x_max'] - tcp_pos[0]) / dt,    # x max workspace limit
            -J_cart[0, :] @ (u + u_nom) <= (tcp_pos[0] - ws['x_min']) / dt,   # x min workspace limit
            J_cart[1, :] @ (u + u_nom) <= (ws['y_max'] - tcp_pos[1]) / dt,    # y max workspace limit
            -J_cart[1, :] @ (u + u_nom) <= (tcp_pos[1] - ws['y_min']) / dt,   # y min workspace limit
            -J_cart[2, :] @ (u + u_nom) <= (tcp_pos[2] - ws['z_min']) / dt,   # z min workspace limit
            ]
        
        problem = cp.Problem(objective, constraints)

        # Naive safety: stop the robot
        u_safe = -u_nom #np.zeros(self.n_joints)
        try:
            problem.solve(solver=cp.OSQP)
            if u.value is None or np.any(np.isnan(u.value)):
                self.loginfo("QP failed")
            else:
                u_safe = u.value
        except Exception as e:
            self.loginfo(f"QP solver error: {e}")

        u = u_nom + u_safe
            
        return u

    def calculate_u_pcbf(self, t, state):
        """
        Calculates the safe velocity command using a QP with the predictive CBF constraint.
        The constraint is: dhstar_dt + dhstar_du·u + α·hstar ≥ 0.
        """
        H, dHdt, dHdu = self.hstar_func(t, state)
        h_val = self.h_func(t, state)
        u_nom = self.mu_func(state)
                
        A = dHdu
        b = - self.alpha * H - dHdt
        
        u = self.solve_qp(u_nom, state, A, b)
            
        return u, h_val
    
    def calculate_u_cbf(self, t, state):
        """
        CBF-based velocity control for single integrator dynamics.
        Enforces: ∇h(q)·u + α·h(q) ≥ 0.
        """
        
        h_val, grad_h_q = self.h_func_with_gradient(t, state)
        u_nom = self.mu_func(state)
        
        A = grad_h_q
        b = -self.lamda * h_val

        u = self.solve_qp(u_nom, state, A, b)
        
        return u, h_val   
    
    def angle_diff(self, a, b):
        diff = a - b
        return np.arctan2(np.sin(diff), np.cos(diff))

    def loginfo(self, msg):
        self.logger.info(str(msg))

def spin_node(node):
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor.spin()

def main():
    rclpy.init()
    node = JAKA()

    #pr = cProfile.Profile()
    #pr.enable()

    if node.publish_robot_state:
        # If we are visualizing, spin the interface node in a separate thread
        interface = node.jaka_interface
        interface_thread = threading.Thread(target=spin_node, args=[interface, ], daemon=True)
        interface_thread.start()
    
    try:
        spin_node(node)
    except KeyboardInterrupt:
        # User pressed Ctrl-C
        node.save_data()
    finally:
        #pr.disable()
        #ps = pstats.Stats(pr).sort_stats('cumtime')
        #ps.print_stats(20)  # top 20 slowest functions
        node.destroy_node()
        rclpy.shutdown()

    if node.publish_robot_state:
        interface_thread.join()
        interface.destroy_node()
    node.destroy_node()

    rclpy.shutdown()

if __name__=='__main__':
    main()