import cvxpy as cp
import numpy as np

from jaka_interface.real_robot import RealRobot
from jaka_interface.pose_conversions import rpy_to_rot_matrix
from scipy.optimize import minimize_scalar

class PredictiveSafeController():
    def __init__(self,
                 robot_instance: RealRobot,
                 cbf_gamma: float = 1,
                 max_joint_velocity: float = 0.1,
                 prediction_horizon: float = 1,
                 dt: float = 0.01):
        self.robot = robot_instance
        
        self.cbf_gamma = cbf_gamma
        self.max_joint_vel = max_joint_velocity

        self.T = prediction_horizon
        
        self.perturb_delta = 0.01
        self.m = 0.001

        self.dt = dt

        self.hand_pos = np.array([0.0, 0.0, 0.0])
        self.hand_radius = 0.1

        self.safety_distance = self.hand_radius

        self.state_len = 6
        self.n_joints = 6

        self.tcp_target = np.array([-0.4, -0.3, 0.3, -np.pi, 0, -20*np.pi/180])
        
        # Control parameters
        self.pos_tolerance = 2e-3
        self.rot_tolerance = 1e-1
        self.pos_gain = 1  # position gain
        self.rot_gain = 1  # orientation gain

    def update(self, t, hand_pos)->np.ndarray:
        """
        Calculates the safe velocity command using a QP with the predictive CBF constraint.
        The constraint is: dhstar_dt + dhstar_du·u + α·hstar ≥ 0.
        """
        
        self.hand_pos = hand_pos if hand_pos is not None else np.zeros(3)

        state = np.array(self.robot.joint_position)
        H, dHdt, dHdu = self.hstar_func(t, state)
        #h_val = self.h_func(t, state)
        u_nom = self.mu_func(state)
        
        J = np.eye(self.n_joints)
        F = np.zeros(self.n_joints)
        
        A = dHdu
        b = - self.cbf_gamma * H - dHdt
        
        u = cp.Variable(self.n_joints)
        objective = cp.Minimize(0.5 * cp.quad_form(u, J) + F @ u)
        constraints = [A @ u <= b,
            u >= -self.max_joint_vel,
            u <= self.max_joint_vel]
        
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP)
            if u.value is None or np.any(np.isnan(u.value)):
                print("QP failed, using nominal control")
                u_safe = u_nom
            else:
                #print(f"{u.value} contro {u_nom} con mano a {hand_pos}")
                u_safe = u.value + u_nom
        except Exception as e:
            print(f"QP solver error: {e}")
            u_safe = u_nom

        return u_safe

    def get_updated_hand(self, t):
        hand_pos = self.hand_pos
        hand_radius = self.hand_radius
            
        return hand_pos, hand_radius

    def h_func(self, t, state):
        """
        Computes the safety margin between a dynamic spherical hand and 
        a cylindrical end-effector (EE) volume centered at the TCP.

        The cylinder is aligned with the TCP z-axis, has height h_cyl, and radius r_cyl.
        """
        # EE shape
        r_cyl = 0.045  # cylinder radius [m]
        h_cyl = -0.35   # cylinder height [m]

        # TCP pose and orientation
        tcp_pose = self.robot.kine_forward(state)
        tcp_pos = tcp_pose.t
        tcp_z_axis = tcp_pose.R[:, 2]  # z-axis of EE

        # Endpoint of cylinder axis
        cyl_top = tcp_pos + h_cyl * tcp_z_axis

        # hand position
        hand_pos, hand_radius = self.get_updated_hand(t)
        self.safety_distance = hand_radius

        # Project hand onto cylinder axis
        axis_vec = cyl_top - tcp_pos
        axis_dir = axis_vec / np.linalg.norm(axis_vec)
        vec_to_obs = hand_pos - tcp_pos
        proj_length = np.dot(vec_to_obs, axis_dir)
        proj_length_clamped = np.clip(proj_length, 0, np.linalg.norm(axis_vec))
        closest_point_on_axis = tcp_pos + proj_length_clamped * axis_dir

        # Distance from hand to closest surface point on the cylinder
        dist_to_cylinder_surface = np.linalg.norm(hand_pos - closest_point_on_axis) - r_cyl

        # Safety margin
        return self.safety_distance - dist_to_cylinder_surface
    
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
        """
        Predicts the state at time tau using single integrator dynamics:
        q_pred = q + u_nom * (tau - t)
        Returns the predicted state, its derivative with respect to tau,
        derivative with respect to t, and the identity derivative w.r.t. state.
        """
        delta_t = tau - t
        u_nom = self.mu_func(state)
        q = state
        q_pred = q + u_nom * delta_t
        
        dp_dtau = u_nom      # ∂q_pred/∂tau
        dp_dt = -u_nom       # ∂q_pred/∂t
        dx = np.eye(self.state_len)  # ∂q_pred/∂q
        
        return q_pred, dp_dtau, dp_dt, dx
    
    def find_max(self, t, state):
        """
        Finds the time tau within [t, t+T] that maximizes the safety margin h.
        Returns tau, the corresponding h value, and an approximate derivative dtau_dx.
        """
                
        # Solve the optimization using trust-constr with the zero Hessian.
        result = minimize_scalar(
            lambda tau: -self.h_func(tau, self.path_func(tau, t, state)[0]),
            bounds=[t, t + self.T],
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
                bounds=[t, t + self.T],
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
    
    def mu_func(self, state):
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