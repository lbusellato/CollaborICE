import numpy as np
import roboticstoolbox as rtb
import cvxpy as cp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

class PCBFSimulation:
    def __init__(self):
        self.dt = 0.05
        self.t = 0

        # Load robot from URDF
        urdf_path = "/home/buse/collaborice_ws/src/jaka_description/urdf/jaka.urdf"
        self.robot = rtb.ERobot.URDF(urdf_path)
        
        # Initial joint configuration (degrees to radians)
        self.robot.q = np.array([85, 89.29, -85.80, 86.51, 90, 40]) * np.pi/180.0
        
        # Joint limits
        #self.q_min = np.array([-170, -80, -170, -170, -170, -360]) * np.pi/180.0
        #self.q_max = np.array([170, 260, 40, 170, 170, 360]) * np.pi/180.0
        
        # Number of joints
        self.n_joints = len(self.robot.q)
        
        # State space is [joint_positions, joint_velocities]
        self.state_len = 2 * self.n_joints
        
        # Set up visualization
        self.sim = rtb.backends.PyPlot.PyPlot()
        self.sim.launch()
        self.sim.add(self.robot)

        # Obstacle definition (in task space)
        self.obstacle_pos = np.array([-0.4, 0.0, 0.3])
        self.obstacle_radius = 0.08
        self.min_dist = 0.1
        self.safety_distance = self.obstacle_radius + self.min_dist

        # PCBF parameters
        self.T = 2.5  # Prediction horizon
        self.m = 4 / (self.T ** 2)  # Barrier function parameter
        self.perturb_delta = 0.1  # For numerical gradient computation
        self.alpha = 1

        # Target TCP pose [x, y, z, r, p, y]
        self.tcp_target = np.array([-0.4, -0.3, 0.3, -np.pi, 0, -20*np.pi/180])
        
        # Control parameters
        self.pos_tolerance = 1e-3
        self.rot_tolerance = 1e-3
        self.pos_gain = 0.1  # position gain
        self.rot_gain = 0.2  # orientation gain
        #self.joint_velocity_limit = 0.5  # rad/s
        
        # Record trajectories for visualization
        self.trajectory = []
        self.h_values = []
        self.control_inputs = []

    def run_simulation(self, max_iterations=400, verbose=False):
        """Run the simulation loop"""
        
        # Initial state: [joint_positions, joint_velocities]
        initial_state = np.concatenate([self.robot.q, np.zeros(self.n_joints)])
        states = [initial_state]
        
        # Main simulation loop
        for i in range(max_iterations):
            # Calculate safe control input
            current_state = states[-1]
            u, h_value = self.calculate_u_pcbf(self.t, current_state)
            
            # Record data for plotting
            self.h_values.append(h_value)
            self.control_inputs.append(u)
            tcp_pose = self.robot.fkine(current_state[:self.n_joints]).t
            self.trajectory.append(tcp_pose)
            
            # Update state
            new_state = self.update_state(self.t, self.t + self.dt, current_state, u)
            states.append(new_state)
            
            # Update visualization
            self.robot.q = new_state[:self.n_joints]
            self.sim.step(self.dt)
            
            # Update time
            self.t += self.dt
            
            # Check for convergence to target
            if self.check_convergence(new_state):
                print(f"Target reached after {i+1} iterations")
                break
            
            # Print progress
            if i % 10 == 0 and verbose:
                print(f"Iteration {i}, t={self.t:.2f}, h={h_value:.4f}")
                    
        
        # Plot results
        self.plot_results(states)
        
        return states, self.h_values, self.control_inputs
    
    def update_state(self, t1, t2, state, u):
        """Update the state using the control input"""
        dt = t2 - t1
        
        # Extract joint positions and velocities
        q = state[:self.n_joints]
        q_dot = state[self.n_joints:]
        
        # Simple double integrator model for joint dynamics
        q_new = q + q_dot * dt
        q_dot_new = q_dot + u * dt
        
        # Apply joint limits
        #q_new = np.clip(q_new, self.q_min, self.q_max)
        
        # Apply velocity limits
        #q_dot_new = np.clip(q_dot_new, -self.joint_velocity_limit, self.joint_velocity_limit)
        
        # Combine into new state
        new_state = np.concatenate([q_new, q_dot_new])
        
        return new_state
    
    def h_func(self, state):
        """
        Safety function: h(x) <= 0 implies safety
        
        Computes the distance between the robot's end effector and the obstacle,
        returns the safety margin (positive when safe)
        """
        # Extract joint positions
        q = state[:self.n_joints]
        
        # Compute forward kinematics to get end effector position
        tcp_pose = self.robot.fkine(q)
        tcp_pos = tcp_pose.t  # Extract translation
        # Compute distance to obstacle
        distance = np.linalg.norm(tcp_pos - self.obstacle_pos)

        h_val = self.safety_distance - distance
        
        # Return safety margin (positive when safe)
        return h_val
    
    def h_func_with_gradient(self, state):
        """Computes the safety function value and its gradient"""
        h_val = self.h_func(state)
        
        # Compute gradient using finite differences
        dh = np.zeros(self.state_len)
        for i in range(self.state_len):
            state_plus = state.copy()
            state_minus = state.copy()
            
            state_plus[i] += self.perturb_delta
            state_minus[i] -= self.perturb_delta
            
            h_plus = self.h_func(state_plus)
            h_minus = self.h_func(state_minus)
            
            dh[i] = (h_plus - h_minus) / (2 * self.perturb_delta)
        
        return h_val, dh
    
    def path_func(self, tau, t, state):
        """
        Predict the state at time tau given current state at time t
        Returns predicted state and its derivatives
        """
        # Time difference
        delta_t = tau - t
        
        # Extract current joint positions and velocities
        q = state[:self.n_joints]
        q_dot = state[self.n_joints:]
        
        # Get nominal control (desired joint accelerations)
        u_nom = self.mu_func(state)
        
        # Predict joint positions and velocities at time tau
        # Using a double integrator model: q(t+Δt) = q(t) + q̇(t)·Δt + 0.5·q̈(t)·Δt²
        q_pred = q + q_dot * delta_t + 0.5 * u_nom * delta_t**2
        q_dot_pred = q_dot + u_nom * delta_t
        
        # Combine into predicted state
        predicted_state = np.concatenate([q_pred, q_dot_pred])
        
        # Compute derivatives if needed (for barrier function)
        if delta_t == 0:
            # At t=tau, the derivatives are identity
            dtau = np.zeros(self.state_len)
            dt = np.zeros(self.state_len)
            dx = np.eye(self.state_len)
        else:
            # Partial derivatives with respect to tau
            dq_dtau = q_dot + u_nom * delta_t
            dq_dot_dtau = u_nom
            dtau = np.concatenate([dq_dtau, dq_dot_dtau])
            
            # Partial derivatives with respect to t (negative of dtau)
            dt = -dtau
            
            # Partial derivatives with respect to x
            dx = np.zeros((self.state_len, self.state_len))
            
            # ∂q_pred/∂q = I
            dx[:self.n_joints, :self.n_joints] = np.eye(self.n_joints)
            
            # ∂q_pred/∂q_dot = Δt·I
            dx[:self.n_joints, self.n_joints:] = np.eye(self.n_joints) * delta_t
            
            # ∂q_dot_pred/∂q_dot = I
            dx[self.n_joints:, self.n_joints:] = np.eye(self.n_joints)
            
            # Note: For full accuracy, we should also compute derivatives of u_nom
            # with respect to state, but we'll omit that for simplicity
        
        return predicted_state, dtau, dt, dx
    
    def find_max(self, t, state):
        """
        Find the time tau that maximizes h(path(tau, t, state))
        Returns the time tau, h(path(tau)), and derivative of tau w.r.t. state
        """
        def objective(tau):
            # Negative because we're minimizing
            predicted_state = self.path_func(tau, t, state)[0]
            return -self.h_func(predicted_state)
        
        # Bound the search to the prediction horizon
        result = minimize_scalar(objective, bounds=(t, t + self.T), method='bounded')
        tau = result.x
        h_of_tau = -result.fun  # Convert back to maximum
        
        # Compute derivative of tau with respect to state
        dtau_dx = np.zeros(self.state_len)
        for i in range(self.state_len):
            state_perturbed = state.copy()
            state_perturbed[i] += self.perturb_delta
            
            # Find new tau with perturbed state
            result_perturbed = minimize_scalar(
                lambda tau: -self.h_func(self.path_func(tau, t, state_perturbed)[0]),
                bounds=(t, t + self.T),
                method='bounded'
            )
            tau_perturbed = result_perturbed.x
            dtau_dx[i] = (tau_perturbed - tau) / self.perturb_delta
        
        return tau, h_of_tau, dtau_dx
    
    def find_zero(self, tau, t, state):
        """
        Find the time eta where h(path(eta, t, state)) = 0
        Returns the time eta and derivative of eta w.r.t. state
        """
        def objective(eta):
            predicted_state = self.path_func(eta, t, state)[0]
            return self.h_func(predicted_state)
        
        # Try to find the zero between t and tau
        try:
            # Use bisection method
            eta = self.bisection_root(objective, t, tau, tol=1e-5)
            
            # Check if the eta is valid
            if eta > tau or np.isnan(eta):
                # Sample points and find closest to zero
                z_values = np.linspace(t, tau, 100)
                h_values = np.array([objective(z) for z in z_values])
                closest_idx = np.argmin(np.abs(h_values))
                start_point = z_values[closest_idx]
                
                # Try again with better starting point
                eta = self.bisection_root(objective, 
                                         max(t, start_point - 0.1), 
                                         min(tau, start_point + 0.1), 
                                         tol=1e-5)
        except Exception as e:
            print(f"Warning: Zero finding failed - {e}")
            eta = t
        
        # Make sure eta is between t and tau
        if eta < t:
            eta = t
        if eta > tau:
            eta = tau
            print("Warning: eta > tau")
        
        # Compute derivative of eta with respect to state
        # Get predicted state at eta
        p, dp_dtau, _, dp_dx = self.path_func(eta, t, state)
        
        # Get gradient of safety function
        _, dh = self.h_func_with_gradient(p)
        
        # Compute derivative (chain rule)
        # deta/dx = -(dh/dp)·(dp/dx) / ((dh/dp)·(dp/dtau))
        denominator = dh @ dp_dtau
        if abs(denominator) < 1e-6:
            # Avoid division by zero
            deta_dx = np.zeros(self.state_len)
        else:
            deta_dx = -(dh @ dp_dx) / denominator
        
        return eta, deta_dx
    
    def bisection_root(self, f, a, b, tol=1e-5, max_iter=100):
        """Find root of function f in interval [a,b] using bisection method"""
        fa = f(a)
        fb = f(b)
        
        # Check if there's a sign change in the interval
        if fa * fb > 0:
            # No sign change, check if close to zero
            if abs(fa) < abs(fb) and abs(fa) < tol:
                return a
            elif abs(fb) < tol:
                return b
            else:
                # Sample interval to look for sign change
                x_vals = np.linspace(a, b, 20)
                f_vals = np.array([f(x) for x in x_vals])
                idx = np.argmin(np.abs(f_vals))
                
                if abs(f_vals[idx]) < tol:
                    return x_vals[idx]
                
                # Check each subinterval
                for i in range(len(x_vals) - 1):
                    if f_vals[i] * f_vals[i+1] <= 0:
                        return self.bisection_root(f, x_vals[i], x_vals[i+1], tol, max_iter)
                
                # If no zero crossing, return endpoint closer to zero
                return a if abs(fa) < abs(fb) else b
        
        # Apply bisection method
        c = a
        fc = fa
        i = 0        
        while (b - a) > tol and i < max_iter:
            c = (a + b) / 2
            fc = f(c)            
            if abs(fc) < tol:
                break            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc            
            i += 1        
        return c
    
    def m_func(self, lambda_val):
        """
        Compute the PCBF margin function m(λ) = c·λ²
        Returns function value and derivative
        """
        m_val = self.m * (lambda_val ** 2)
        dm_val = 2 * self.m * lambda_val
        return m_val, dm_val
    
    def hstar_func(self, t, state):
        """
        Compute the predictive CBF value and its derivatives
        Returns hstar, dhstar/dt, and dhstar/du
        """
        # Find time of maximum safety violation
        m1star, h_of_m1star, dm1star_dx = self.find_max(t, state)
        
        # Determine if we need to find zero crossing
        if h_of_m1star <= 0:
            # First local maxima is safe, use it
            r = m1star
            dr_dx = dm1star_dx
        else:
            # Find zero crossing time
            r, dr_dx = self.find_zero(m1star, t, state)
            
            # Switch to M1star if derivative gets too large
            if np.linalg.norm(dr_dx) >= 10 * np.linalg.norm(dm1star_dx):
                dr_dx = dm1star_dx
        
        # Compute margin function
        m_val, dm_val = self.m_func(r - t)
        
        # Compute H*
        hstar = h_of_m1star - m_val
        
        # Get predicted state at m1star
        predicted_state, _, _, dp_of_tau_dx = self.path_func(m1star, t, state)
        
        # Get gradient of safety function at predicted state
        _, dh_of_tau_dx = self.h_func_with_gradient(predicted_state)
        
        # Control influence matrix (how control affects state)
        g = np.zeros((self.state_len, self.n_joints))
        g[self.n_joints:, :] = np.eye(self.n_joints)  # Control only affects velocities
        
        # Compute derivatives for control computation
        if m1star <= t + 1e-4:
            # Case iii - local maximum is at current time
            # Compute numerical derivative of h w.r.t. tau
            path_plus_delta = self.path_func(m1star + self.perturb_delta, t, state)[0]
            h_plus_delta = self.h_func(path_plus_delta)
            dh_dtau = (h_plus_delta - h_of_m1star) / self.perturb_delta
            
            # Since the system is control-affine, dtau_dt = 1
            dhstar_dt = dh_dtau
            dhstar_du = (dh_of_tau_dx @ dp_of_tau_dx @ g)
            
        elif m1star < t + self.T:
            # Case i - local maximum is within prediction horizon
            dhstar_dt = -dm_val  # Time derivative comes from margin function
            dhstar_du = (dh_of_tau_dx @ dp_of_tau_dx - dm_val * dr_dx) @ g
            
        elif m1star <= t + self.T + 1e-4:
            # Case ii/iii - local maximum is at prediction horizon boundary
            if h_of_m1star > 0:
                # Case ii - unsafe at horizon boundary
                path_plus_delta = self.path_func(m1star + self.perturb_delta, t, state)[0]
                h_plus_delta = self.h_func(path_plus_delta)
                dh_dtau = (h_plus_delta - h_of_m1star) / self.perturb_delta
                
                dhstar_dt = -dm_val + dh_dtau
                dhstar_du = (dh_of_tau_dx @ dp_of_tau_dx - dm_val * dr_dx) @ g
            else:
                # Case iii - safe at horizon boundary
                path_plus_delta = self.path_func(m1star + self.perturb_delta, t, state)[0]
                h_plus_delta = self.h_func(path_plus_delta)
                dh_dtau = (h_plus_delta - h_of_m1star) / self.perturb_delta
                
                dhstar_dt = dh_dtau
                dhstar_du = dh_of_tau_dx @ dp_of_tau_dx @ g
        else:
            print("Warning: m1star > t+T")
            dhstar_dt = 0
            dhstar_du = np.zeros(self.n_joints)
        
        return hstar, dhstar_dt, dhstar_du
    
    def mu_func(self, state):
        """
        Nominal control law (task-space PD controller)
        Computes desired joint accelerations to reach target
        """
        # Extract joint positions and velocities
        q = state[:self.n_joints]
        q_dot = state[self.n_joints:]
        
        # Get current end-effector pose
        current_pose = self.robot.fkine(q)
        current_pos = current_pose.t
        
        # Get target position
        target_pos = self.tcp_target[:3]
        
        # Position error
        pos_error = target_pos - current_pos
        pos_error_mag = np.linalg.norm(pos_error)

        rot_current = self.rpy_to_R(current_pose.rpy())
        rot_target = self.rpy_to_R(self.tcp_target[3:6])

        rot_error = rot_target.dot(rot_current.T)
        rot_error = 0.5 * np.array([
            rot_error[2, 1] - rot_error[1, 2],
            rot_error[0, 2] - rot_error[2, 0],
            rot_error[1, 0] - rot_error[0, 1]
        ])
        rot_error_norm = np.linalg.norm(rot_error)
                    
        # Compute task-space velocity command
        v_cmd = np.zeros(6)
        
        # Position component
        if pos_error_mag > self.pos_tolerance:
            v_cmd[:3] = self.pos_gain * pos_error / max(pos_error_mag, 1e-6)
        
        # Rotation component
        if rot_error_norm > self.rot_tolerance:
            v_cmd[3:] = self.rot_gain * rot_error / max(rot_error_norm, 1e-6)
            
        # Get Jacobian
        J = self.robot.jacob0(q)
        
        # Compute pseudo-inverse of Jacobian
        J_pinv = np.linalg.pinv(J)
        
        # Transform to joint velocities
        q_dot_des = J_pinv @ v_cmd
        
        # Simple PD control for joint accelerations
        kp = 5.0  # Position gain
        kd = 1.0  # Damping gain
        
        # q̈ = kp·(q̇_des - q̇) - kd·q̇
        u = kp * (q_dot_des - q_dot) - kd * q_dot
        
        return u
    
    def calculate_u_pcbf(self, t, state):
        """
        Calculate control input using QP with PCBF constraint
        Returns the safe control input and the safety value
        """
        # Get barrier function and derivatives
        hstar, dhstar_dt, dhstar_du = self.hstar_func(t, state)
        
        # Get nominal control
        u_nom = self.mu_func(state)
                
        # Set up QP
        # Objective: minimize ||u - u_nom||²
        J = np.eye(self.n_joints)  # Cost matrix
        F = -u_nom  # Linear term
        
        # Constraint: dhstar_dt + dhstar_du · u + α · hstar ≥ 0
        # Rearranged: dhstar_du · u ≥ -dhstar_dt - α · hstar
        A = dhstar_du.reshape(1, self.n_joints)
        b = np.array([-dhstar_dt - self.alpha * hstar])
        
        # Solve QP using CVXPY
        u = cp.Variable(self.n_joints)
        objective = cp.Minimize(0.5 * cp.quad_form(u, J) + F.T @ u)
        constraints = [A @ u >= b]
        
        # Add joint acceleration limits if needed
        #constraints.extend([
        #    u >= -5.0,  # Lower limit
        #    u <= 5.0,   # Upper limit
        #])
        
        # Solve the problem
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.OSQP)
            
            if u.value is None or np.any(np.isnan(u.value)):
                print("QP failed, using nominal control")
                u_safe = u_nom
            else:
                u_safe = u.value
        except Exception as e:
            print(f"QP solver error: {e}")
            u_safe = u_nom
        
        return u_safe, hstar
    
    def check_convergence(self, state):
        """Check if the robot has reached the target"""
        # Get current end-effector pose
        q = state[:self.n_joints]
        current_pose = self.robot.fkine(q)
        current_pos = current_pose.t
        
        # Position error
        pos_error = np.linalg.norm(current_pos - self.tcp_target[:3])
        
        # Orientation error (simplified)
        current_rpy = self.extract_rpy(current_pose)
        rot_error = np.linalg.norm(current_rpy - self.tcp_target[3:6])
        
        # Check if within tolerance
        return (pos_error < self.pos_tolerance * 10 and 
                rot_error < self.rot_tolerance * 10)
    
    def extract_rpy(self, pose):
        """Extract RPY angles from pose matrix"""
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
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_results(self, states):
        """Plot simulation results"""
        
        # Time vector
        t_vec = np.arange(0, len(states) * self.dt, self.dt)
                
        # Plot safety value
        plt.subplot(1, 2, 1)
        plt.plot(t_vec[:-1], self.h_values, label='h')
        plt.axhline(y=0, color='b', linestyle='--', label='Safety Threshold')
        plt.axhline(y=self.safety_distance, color='r', linestyle='--', label='Safety Distance')
        plt.xlabel('Time (s)')
        plt.ylabel('Safety Value')
        plt.legend()
        plt.title('Safety Constraint (h* ≥ 0 is unsafe)')
        plt.grid(True)
        
        # Plot end-effector trajectory in 3D
        plt.subplot(1, 2, 2, projection='3d')
        traj = np.array(self.trajectory)
        ax = plt.gca()
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', label='EE Path')
        ax.scatter(self.obstacle_pos[0], self.obstacle_pos[1], self.obstacle_pos[2], 
                   color='r', s=100, label='Obstacle')
        ax.scatter(self.tcp_target[0], self.tcp_target[1], self.tcp_target[2], 
                   color='g', s=100, label='Target')
        
        
        # Draw sphere around obstacle
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.obstacle_pos[0] + self.obstacle_radius * np.cos(u) * np.sin(v)
        y = self.obstacle_pos[1] + self.obstacle_radius * np.sin(u) * np.sin(v)
        z = self.obstacle_pos[2] + self.obstacle_radius * np.cos(v)
        ax.plot_surface(x, y, z, color='r', alpha=0.2)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('End-Effector Trajectory')
        ax.legend()
        self.set_axes_equal(ax)
        
        plt.tight_layout()
        plt.show(block=True)


def main():
    sim = PCBFSimulation()
    sim.run_simulation(max_iterations=300)

if __name__ == "__main__":
    main()