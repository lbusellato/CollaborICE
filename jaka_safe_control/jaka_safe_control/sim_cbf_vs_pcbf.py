import numpy as np
import roboticstoolbox as rtb
import cvxpy as cp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import tqdm
from roboticstoolbox.backends.swift import Swift
from roboticstoolbox.backends.PyPlot import PyPlot
import spatialgeometry as sg
import spatialmath as sm    
import time
import math

VISUAL_SIM = True
DO_PREDICTIVE = True
HAND = True
LINEAR_PRED = True

def rpy_to_rot_matrix(rpy: list) -> list:
    """Convert RPY angles to a rotation matrix.

    Parameters
    ----------
    rpy : list
        The xyz roll, pitch and yaw.

    Returns
    -------
    list
        The corresponding ZYX rotation matrix
    """
    roll, pitch, yaw = rpy
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    # Compose the rotation matrix R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.array([
         [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
         [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
         [-sp,   cp*sr,            cp*cr]
        ])
    return R

class PCBFSimulation:
    def __init__(self):
        self.dt = 0.05
        self.t = 0

        # Load robot from URDF
        urdf_path = "/home/jaka/ros2_ws/src/jaka_description/urdf/jaka_swift.urdf"
        self.robot = rtb.ERobot.URDF(urdf_path)
        
        # Initial joint configuration (degrees to radians)
        self.robot.q = np.array([85, 89.29, -85.80, 86.51, 90, 40]) * np.pi/180.0
        self.initial_tcp = self.robot.fkine(self.robot.q).t
        
        # Number of joints
        self.n_joints = len(self.robot.q)
        
        # Now the state is just joint positions (single integrator)
        self.state_len = self.n_joints

        self.leap_hand = np.load('./hand_pos.npy')           # shape (N, 3)
        self.leap_radius = np.load('./hand_radius.npy').ravel()  # shape (N,)
        self.leap_start_count = 1600
        self.step_count = self.leap_start_count

        # Obstacle definition (in task space)
        self.obstacle_pos = self.leap_hand[self.step_count]
        self.obstacle_radius = 0.05 # For sim purposes, can't dinamically change the radius of the plotted sphere
        self.safety_distance = self.leap_radius[self.step_count]

        self.obstacle_base_pos = self.obstacle_pos.copy()  
        self.obstacle_amplitude = 0.05    # meters
        self.obstacle_freq      = 0.25    # Hz (i.e. one full up-down every 2s)

        # Workspace definition
        self.workspace_limits = {
            'x_min': -0.5, 'x_max': 0.1, # don't hit the operator/camera rig
            'y_min': -0.5, 'y_max': 0.7, # don't hit the milling
            'z_min':  0.3           # don't go below table height
        }

        self.cyl_radius = 0.045
        self.cyl_height = 0.35

        # Set up visualization
        if VISUAL_SIM:
            self.sim = Swift()
            
            self.sim.launch()
            self.sim.add(self.robot)

            self.obstacle_sphere = sg.Sphere(
                radius=self.obstacle_radius, 
                pose=sm.SE3(self.obstacle_pos), 
                color=[0, 0, 1, 0.5]  # RGBA, 50% transparent red
            )


            self.sim.add(self.obstacle_sphere)
            
            # add EE cylinder (initial pose at origin)
            self.ee_cyl_vis = sg.Cylinder(
                radius=self.cyl_radius, 
                length=self.cyl_height, 
                color=[0.0,1.0,0.0,0.3])

            self.sim.add(self.ee_cyl_vis)
            
            self.sim.step()

        # PCBF parameters
        self.T = 1  # Prediction horizon
        self.m = 1e-4  # Barrier function parameter
        self.perturb_delta = 0.01  # For numerical gradient computation
        self.alpha = 40
        self.lamda = self.alpha # for consistency, static = 0.8, dynamic = 40, predictive = 40
        self.max_joint_vel = np.pi / 2

        self.smoothing = 0.5
        self.prev_u = np.zeros(6)

        # Target TCP pose [x, y, z, r, p, y]
        self.tcp_target = np.array([-0.4, -0.3, 0.3, -np.pi, 0, -20*np.pi/180])

        # Control parameters
        self.pos_tolerance = 2e-3
        self.rot_tolerance = 1e-1
        self.pos_gain = 0.1  # position gain
        self.rot_gain = 0.1  # orientation gain

        # Record trajectories for visualization
        self.trajectory = []
        self.h_values = []
        self.control_inputs = []

    def run_simulation(self, method='vanilla'):
        """Run the simulation loop using velocity control."""
        if method == 'vanilla':
            control_method = self.calculate_u_cbf
        elif method == 'predictive':
            control_method = self.calculate_u_pcbf

        # Initial state is the joint positions
        initial_state = self.robot.q.copy()
        states = [initial_state]
        
        max_it = 1000
        for i in tqdm.tqdm(range(max_it), desc=f"{method} CBF"):
            start = time.time()
            
            obstacle_pos, obstacle_radius = self.get_updated_obstacle(self.t)
            self.obstacle_pos = obstacle_pos
            self.safety_distance = obstacle_radius

            if self.check_convergence(states[-1]): break

            current_state = states[-1]                            
            u, h_value = control_method(self.t, current_state)   

            u = self.smoothing * u + (1 - self.smoothing) * self.prev_u
            self.prev_u = u

            self.h_values.append(h_value)
            self.control_inputs.append(u)
            tcp_pose = self.robot.fkine(current_state).t
            self.trajectory.append(tcp_pose)
            
            # Update state using single integrator dynamics: q_new = q + u*dt
            new_state = current_state + u * self.dt
            states.append(new_state)            
            self.robot.q = new_state

            if VISUAL_SIM: 

                self.obstacle_sphere.T = sm.SE3(self.obstacle_pos)
                # use the *new* state, and take the full SE3 (rotation+translation)
                offset = sm.SE3(0, 0, -self.cyl_height/2.0)
                
                ee_pose = self.robot.fkine(self.robot.q)        # this is an SE3 object already
                self.ee_cyl_vis.T = ee_pose * offset                    # preserves both R and t
                self.sim.step(self.dt)
            
            self.step_count += 1
            self.t += self.dt

            # This aligns timings between different simulations (or tries to at least)
            actual_dt = time.time() - start
            if actual_dt < self.dt: time.sleep(self.dt - actual_dt)
                            
        return states, self.h_values, np.vstack(self.control_inputs)

    def get_updated_obstacle(self, t):
        if HAND:
            idx = self.step_count + round((t - self.t) / self.dt)

            if LINEAR_PRED and idx >= 1:
                # Linear prediction based on current and previous position
                prev_idx = max(idx - 1, 0)
                prev_pos = self.leap_hand[prev_idx]
                curr_pos = self.leap_hand[idx]
                velocity = (curr_pos - prev_pos) / self.dt
                dt_future = (t - self.t)
                obstacle_pos = curr_pos + velocity * dt_future
            else:
                obstacle_pos = self.leap_hand[idx]
            
            obstacle_radius = self.leap_radius[min(idx, len(self.leap_radius) - 1)]

        else:
            obstacle_pos = self.obstacle_pos.copy()
            obstacle_pos[2] = (
                self.obstacle_base_pos[2]
                + self.obstacle_amplitude 
                * np.sin(2*np.pi * self.obstacle_freq * t)
            )
            obstacle_radius = self.obstacle_radius
            
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
        tcp_pose = self.robot.fkine(state)
        tcp_pos = tcp_pose.t
        tcp_z_axis = tcp_pose.R[:, 2]  # z-axis of EE

        # Endpoint of cylinder axis
        cyl_top = tcp_pos - h_cyl * tcp_z_axis

        # Obstacle position
        obstacle_pos, obstacle_radius = self.get_updated_obstacle(t)
        self.safety_distance = obstacle_radius

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
        current_pose = self.robot.fkine(q)
        current_pos = current_pose.t
        target_pos = self.tcp_target[:3]
        pos_error = target_pos - current_pos
        pos_error_mag = np.linalg.norm(pos_error)

        rot_current = self.rpy_to_R(current_pose.rpy())
        rot_target = self.rpy_to_R(self.tcp_target[3:6])
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
            
        J = self.robot.jacob0(q)
        J_pinv = np.linalg.pinv(J)
        q_dot_des = J_pinv @ v_cmd
        
        return q_dot_des
    
    def solve_qp(self, u_nom, q, A, b, P=np.eye(6), F=np.zeros(6)):
        
        u = cp.Variable(self.n_joints)

        objective = cp.Minimize(0.5 * cp.quad_form(u, P) + F @ u)

        J_cart = self.robot.jacob0(q)
        tcp_pos = self.robot.fkine(q).t


        ws = self.workspace_limits
        dt = self.dt

        constraints = [
            A @ u <= b,                                             # CBF constraint
            u >= -self.max_joint_vel,                               # max velocity constraint
            u <= self.max_joint_vel,                                # max velocity constraint
            J_cart[0, :] @ u <= (ws['x_max'] - tcp_pos[0]) / dt,    # x max workspace limit
            -J_cart[0, :] @ u <= (tcp_pos[0] - ws['x_min']) / dt,   # x min workspace limit
            J_cart[1, :] @ u <= (ws['y_max'] - tcp_pos[1]) / dt,    # y max workspace limit
            -J_cart[1, :] @ u <= (tcp_pos[1] - ws['y_min']) / dt,   # y min workspace limit
            -J_cart[2, :] @ u <= (tcp_pos[2] - ws['z_min']) / dt,   # z min workspace limit
            ]
        
        problem = cp.Problem(objective, constraints)

        # Naive safety: stop the robot
        u_safe = -u_nom #np.zeros(self.n_joints)
        try:
            problem.solve(solver=cp.OSQP)
            if u.value is None or np.any(np.isnan(u.value)):
                print("QP failed")
            else:
                u_safe = u.value
        except Exception as e:
            print(f"QP solver error: {e}")

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
    
    def check_convergence(self, state):
        """Checks if the end-effector is close enough to the target."""
        q = state
        current_pose = self.robot.fkine(q)
        current_pos = current_pose.t
        pos_error = np.linalg.norm(current_pos - self.tcp_target[:3])
        current_rpy = self.extract_rpy(current_pose)
        rpy_error = np.array([self.angle_diff(current_rpy[i], self.tcp_target[3+i]) for i in range(3)])
        rot_error = np.linalg.norm(rpy_error)
        
        return (pos_error < self.pos_tolerance * 10 and 
                rot_error < self.rot_tolerance * 10)
    
    def angle_diff(self, a, b):
        diff = a - b
        return np.arctan2(np.sin(diff), np.cos(diff))

    def extract_rpy(self, pose):
        """Extract RPY angles from the pose matrix."""
        return np.array(pose.rpy())
    
    def rpy_to_R(self, rpy):
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ])
        return R
    
    def set_axes_equal(self, ax):
        """Ensures equal scaling on all axes for a 3D plot."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_results(self, states):
        """Plots safety value and end-effector trajectory."""
        plt.subplot(1, 2, 1)
        plt.plot(self.h_values, label='h')
        plt.axhline(y=0, color='b', linestyle='--', label='Safety Threshold')
        plt.xlabel('Time (s)')
        plt.ylabel('Safety Value')
        plt.legend()
        plt.title('Safety Constraint (h* ≥ 0 is unsafe)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2, projection='3d')
        traj = np.array(self.trajectory)
        ax = plt.gca()
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', label='EE Path')
        ax.scatter(self.obstacle_pos[0], self.obstacle_pos[1], self.obstacle_pos[2], 
                   color='r', s=100, label='Obstacle')
        ax.scatter(self.tcp_target[0], self.tcp_target[1], self.tcp_target[2], 
                   color='g', s=100, label='Target')
        ax.scatter(self.initial_tcp[0], self.initial_tcp[1], self.initial_tcp[2], 
                   color='b', s=100, label='Start')
        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.obstacle_pos[0] + self.safety_distance * np.cos(u) * np.sin(v)
        y = self.obstacle_pos[1] + self.safety_distance * np.sin(u) * np.sin(v)
        z = self.obstacle_pos[2] + self.safety_distance * np.cos(v)
        ax.plot_surface(x, y, z, color='r', alpha=0.2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('End-Effector Trajectory')
        ax.legend()
        self.set_axes_equal(ax)
        
        plt.tight_layout()
        plt.show(block=True)

def compare_simulations():

    # Run simulation with predictive CBF
    if DO_PREDICTIVE: sim_predictive = PCBFSimulation()
    if DO_PREDICTIVE: states_predictive, h_vals_predictive, u_predictive = sim_predictive.run_simulation(method='predictive')

    # Run simulation with vanilla (standard) CBF
    sim_vanilla = PCBFSimulation()
    states_vanilla, h_vals_vanilla, u_vanilla = sim_vanilla.run_simulation(method='vanilla')
    
    if VISUAL_SIM:
        sim_vanilla.sim.close()
        if DO_PREDICTIVE: sim_predictive.sim.close()

    # Plot safety value comparison
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(121)
    ax.plot(h_vals_vanilla, label="Vanilla CBF")
    if DO_PREDICTIVE: ax.plot(h_vals_predictive, label="Predictive CBF")
    ax.axhline(y=0, color='black', linestyle='--', label="Safety Threshold")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Safety Value")
    ax.set_title("Safety Value Comparison")
    ax.legend()
    ax.grid(True)

    ax = fig.add_subplot(122)
    ax.plot(u_vanilla[:,0], label="Vanilla CBF")
    ax.plot(u_predictive[:,0], label="Predictive CBF")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Velocity [rad/s]")
    ax.set_title("Joint 0 control signal Comparison")
    ax.legend()
    ax.grid(True)
    
    # Plot end-effector trajectory comparison in 3D
    #fig = plt.figure(figsize=(10,8))
    #ax = fig.add_subplot(111, projection='3d')
    
    traj_vanilla = np.array([sim_vanilla.robot.fkine(state).t for state in states_vanilla])
    #if DO_PREDICTIVE: traj_predictive = np.array([sim_predictive.robot.fkine(state).t for state in states_predictive])
    
    #ax.scatter(traj_vanilla[:, 0], traj_vanilla[:, 1], traj_vanilla[:, 2], label="Vanilla CBF", color='blue')
    #if DO_PREDICTIVE: ax.scatter(traj_predictive[:, 0], traj_predictive[:, 1], traj_predictive[:, 2], label="Predictive CBF", color='orange')
    
    obstacle_pos = sim_vanilla.obstacle_pos
    tcp_target = sim_vanilla.tcp_target
    safety_distance = sim_vanilla.safety_distance
    initial_tcp = sim_vanilla.initial_tcp

    #ax.scatter(obstacle_pos[0], obstacle_pos[1], obstacle_pos[2],
    #        color='red', s=100, label="Obstacle")
    #ax.scatter(tcp_target[0], tcp_target[1], tcp_target[2],
    #        color='green', s=100, label="Target")
    #ax.scatter(initial_tcp[0], initial_tcp[1], initial_tcp[2],
    #        color='blue', s=100, label="Start")
    
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = obstacle_pos[0] + safety_distance * np.cos(u) * np.sin(v)
    y = obstacle_pos[1] + safety_distance * np.sin(u) * np.sin(v)
    z = obstacle_pos[2] + safety_distance * np.cos(v)
    #ax.plot_surface(x, y, z, color='r', alpha=0.2)
    
    #ax.set_xlabel("X (m)")
    #ax.set_ylabel("Y (m)")
    #ax.set_zlabel("Z (m)")
    #ax.set_title("End-Effector Trajectory Comparison")
    #ax.legend()

    #sim_vanilla.set_axes_equal(ax)
    
    plt.show(block=True)

def main():
    compare_simulations()

if __name__ == "__main__":
    main()