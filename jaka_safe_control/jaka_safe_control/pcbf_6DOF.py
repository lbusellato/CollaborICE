import numpy as np
import roboticstoolbox as rtb
import cvxpy as cp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import time

VISUAL_SIM = False

class PCBFSimulation:
    def __init__(self):
        self.dt = 0.1
        self.t = 0

        # Load robot from URDF
        urdf_path = "/home/buse/collaborice_ws/src/jaka_description/urdf/jaka.urdf"
        self.robot = rtb.ERobot.URDF(urdf_path)
        
        # Initial joint configuration (degrees to radians)
        self.robot.q = np.array([85, 89.29, -85.80, 86.51, 90, 40]) * np.pi/180.0
        self.initial_tcp = self.robot.fkine(self.robot.q).t
        
        # Number of joints
        self.n_joints = len(self.robot.q)
        
        # Now the state is just joint positions (single integrator)
        self.state_len = self.n_joints
        
        # Set up visualization
        if VISUAL_SIM:
            self.sim = rtb.backends.PyPlot.PyPlot()
            self.sim.launch()
            self.sim.add(self.robot)

        # Obstacle definition (in task space)
        self.obstacle_pos = np.array([-0.4, 0.0, 0.3])
        self.obstacle_radius = 0.08
        self.safety_distance = self.obstacle_radius

        # PCBF parameters
        self.T = 3  # Prediction horizon
        self.m = 6 / (self.T ** 2)  # Barrier function parameter
        self.perturb_delta = 0.1  # For numerical gradient computation
        self.alpha = 1
        self.lamda = 0.8
        self.max_joint_vel = np.pi / 2

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
        
        it = 0
        max_it = 1000
        while not (self.check_convergence(states[-1]) or it >= max_it):
            current_state = states[-1]
                            
            u, h_value = control_method(self.t, current_state)

            self.h_values.append(h_value)
            self.control_inputs.append(u)
            tcp_pose = self.robot.fkine(current_state).t
            self.trajectory.append(tcp_pose)
            
            # Update state using single integrator dynamics: q_new = q + u*dt
            new_state = current_state + u * self.dt
            states.append(new_state)
            
            self.robot.q = new_state
            if VISUAL_SIM:
                self.sim.step(self.dt)
            
            self.t += self.dt
            it += 1
                
        return states, self.h_values, np.vstack(self.control_inputs)
    
    def h_func(self, state):
        """
        Computes the safety margin: the distance between the end-effector
        and the obstacle minus the safety distance.
        """
        tcp_pos = self.robot.fkine(state).t
        distance = np.linalg.norm(tcp_pos - self.obstacle_pos)
        h_val = distance - self.safety_distance
        return h_val
    
    def h_func_with_gradient(self, state):
        """Numerically computes the gradient of h with respect to the state q."""
        h_val = self.h_func(state)
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
        Returns tau, the corresponding h value, and an approximate derivative.
        """
        def objective(tau):
            q_pred = self.path_func(tau, t, state)[0]
            return -self.h_func(q_pred)
        result = minimize_scalar(objective, bounds=(t, t + self.T), method='bounded')
        tau = result.x
        h_of_tau = -result.fun
        dtau_dx = np.zeros(self.state_len)
        for i in range(self.state_len):
            state_perturbed = state.copy()
            state_perturbed[i] += self.perturb_delta
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
        Finds the time eta where the predicted safety margin crosses zero.
        Returns eta and its derivative approximation.
        """
        def objective(eta):
            q_pred = self.path_func(eta, t, state)[0]
            return self.h_func(q_pred)
        try:
            eta = self.bisection_root(objective, t, tau, tol=1e-5)
            if eta > tau or np.isnan(eta):
                z_values = np.linspace(t, tau, 100)
                h_values = np.array([objective(z) for z in z_values])
                closest_idx = np.argmin(np.abs(h_values))
                start_point = z_values[closest_idx]
                eta = self.bisection_root(objective, 
                                         max(t, start_point - 0.1), 
                                         min(tau, start_point + 0.1), 
                                         tol=1e-5)
        except Exception as e:
            print(f"Warning: Zero finding failed - {e}")
            eta = t
        
        if eta < t:
            eta = t
        if eta > tau:
            eta = tau
            print("Warning: eta > tau")
        
        p, dp_dtau, _, dp_dx = self.path_func(eta, t, state)
        _, dh = self.h_func_with_gradient(p)
        denominator = dh.dot(dp_dtau)
        if abs(denominator) < 1e-6:
            deta_dx = np.zeros(self.state_len)
        else:
            deta_dx = -(dh * dp_dx.diagonal()) / denominator  # approximate derivative
        return eta, deta_dx
    
    def bisection_root(self, f, a, b, tol=1e-5, max_iter=100):
        """A simple bisection method to find a root of f in [a, b]."""
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            if abs(fa) < abs(fb) and abs(fa) < tol:
                return a
            elif abs(fb) < tol:
                return b
            else:
                x_vals = np.linspace(a, b, 20)
                f_vals = np.array([f(x) for x in x_vals])
                idx = np.argmin(np.abs(f_vals))
                if abs(f_vals[idx]) < tol:
                    return x_vals[idx]
                for i in range(len(x_vals) - 1):
                    if f_vals[i] * f_vals[i+1] <= 0:
                        return self.bisection_root(f, x_vals[i], x_vals[i+1], tol, max_iter)
                return a if abs(fa) < abs(fb) else b
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
        m_val = self.m * (lambda_val ** 2)
        dm_val = 2 * self.m * lambda_val
        return m_val, dm_val
    
    def hstar_func(self, t, state):
        """
        Computes the predictive CBF value h* and its derivatives
        for the single integrator (velocity control) case.
        """
        m1star, h_of_m1star, dm1star_dx = self.find_max(t, state)
        if h_of_m1star >= 0:
            r = m1star
            dr_dx = dm1star_dx
        else:
            r, dr_dx = self.find_zero(m1star, t, state)
            print('ye')
            if np.linalg.norm(dr_dx) >= 10 * np.linalg.norm(dm1star_dx):
                dr_dx = dm1star_dx
        m_val, dm_val = self.m_func(r - t)
        hstar = h_of_m1star - m_val
        predicted_state, dp_dtau, _, dp_dx = self.path_func(m1star, t, state)
        _, dh_of_tau_dx = self.h_func_with_gradient(predicted_state)
                
        if m1star <= t + 1e-4:
            q_pred_plus = self.path_func(m1star + self.perturb_delta, t, state)[0]
            h_plus_delta = self.h_func(q_pred_plus)
            dh_dtau = (h_plus_delta - h_of_m1star) / self.perturb_delta
            dhstar_dt = dh_dtau
            dhstar_du = dh_of_tau_dx
        elif m1star < t + self.T:
            dhstar_dt = -dm_val
            dhstar_du = (dh_of_tau_dx - dm_val * dr_dx)
        elif m1star <= t + self.T + 1e-4:
            if h_of_m1star > 0:
                q_pred_plus = self.path_func(m1star + self.perturb_delta, t, state)[0]
                h_plus_delta = self.h_func(q_pred_plus)
                dh_dtau = (h_plus_delta - h_of_m1star) / self.perturb_delta
                dhstar_dt = -dm_val + dh_dtau
                dhstar_du = (dh_of_tau_dx - dm_val * dr_dx)
            else:
                q_pred_plus = self.path_func(m1star + self.perturb_delta, t, state)[0]
                h_plus_delta = self.h_func(q_pred_plus)
                dh_dtau = (h_plus_delta - h_of_m1star) / self.perturb_delta
                dhstar_dt = dh_dtau
                dhstar_du = dh_of_tau_dx
        else:
            print("Warning: m1star > t+T")
            dhstar_dt = 0
            dhstar_du = np.zeros(self.state_len)
        
        return hstar, dhstar_dt, dhstar_du
    
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
    
    def calculate_u_pcbf(self, t, state):
        """
        Calculates the safe velocity command using a QP with the predictive CBF constraint.
        The constraint is: dhstar_dt + dhstar_du·u + α·hstar ≥ 0.
        """
        hstar, dhstar_dt, dhstar_du = self.hstar_func(t, state)
        h_val = self.h_func(state)
        u_nom = self.mu_func(state)
                
        J = np.eye(self.n_joints)
        F = -u_nom
        
        A = dhstar_du.reshape(1, self.n_joints)
        b = np.array([-dhstar_dt - self.alpha * hstar])
        
        u = cp.Variable(self.n_joints)
        objective = cp.Minimize(0.5 * cp.quad_form(u, J) + F.T @ u)
        constraints = [A @ u >= b,
            u >= -self.max_joint_vel,
            u <= self.max_joint_vel]
        
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
            
        return u_safe, h_val
    
    def calculate_u_cbf(self, t, state):
        """
        CBF-based velocity control for single integrator dynamics.
        Enforces: ∇h(q)·u + α·h(q) ≥ 0.
        """
        h_val = self.h_func(state)
        _, grad_h_q = self.h_func_with_gradient(state)
        u_nom = self.mu_func(state)
        
        b_rhs = -self.lamda * h_val
        
        J = np.eye(self.n_joints)
        F = -u_nom
        
        u = cp.Variable(self.n_joints)
        objective = cp.Minimize(0.5 * cp.quad_form(u, J) + F.T @ u)
        constraints = [grad_h_q.reshape(1, self.n_joints) @ u >= b_rhs]
        
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
        
        return u_safe, h_val
    
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

    # Run simulation with vanilla (standard) CBF
    print('Simulating vanilla CBF...', end='')
    start_time = time.time()
    sim_vanilla = PCBFSimulation()
    states_vanilla, h_vals_vanilla, u_vanilla = sim_vanilla.run_simulation(method='vanilla')
    print('done.')
    elapsed = time.time() - start_time
    avg_dt = elapsed / len(states_vanilla)
    print(f"Time elapsed: {elapsed},  avg timestep: {avg_dt}")

    # Run simulation with predictive CBF
    print('Simulating predictive CBF...', end='')
    start_time = time.time()
    sim_predictive = PCBFSimulation()
    states_predictive, h_vals_predictive, u_predictive = sim_predictive.run_simulation(method='predictive')
    print('done.')
    elapsed = time.time() - start_time
    avg_dt = elapsed / len(states_predictive)
    print(f"Time elapsed: {elapsed},  avg timestep: {avg_dt}")
    
    if VISUAL_SIM:
        sim_vanilla.sim.close()
        sim_predictive.sim.close()

    # Plot safety value comparison
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(121)
    ax.plot(h_vals_vanilla, label="Vanilla CBF")
    ax.plot(h_vals_predictive, label="Predictive CBF")
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
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    
    traj_vanilla = np.array([sim_vanilla.robot.fkine(state).t for state in states_vanilla])
    traj_predictive = np.array([sim_predictive.robot.fkine(state).t for state in states_predictive])
    
    ax.scatter(traj_vanilla[:, 0], traj_vanilla[:, 1], traj_vanilla[:, 2], label="Vanilla CBF", color='blue')
    ax.scatter(traj_predictive[:, 0], traj_predictive[:, 1], traj_predictive[:, 2], label="Predictive CBF", color='orange')
    
    ax.scatter(sim_predictive.obstacle_pos[0], sim_predictive.obstacle_pos[1], sim_predictive.obstacle_pos[2],
            color='red', s=100, label="Obstacle")
    ax.scatter(sim_predictive.tcp_target[0], sim_predictive.tcp_target[1], sim_predictive.tcp_target[2],
            color='green', s=100, label="Target")
    ax.scatter(sim_predictive.initial_tcp[0], sim_predictive.initial_tcp[1], sim_predictive.initial_tcp[2],
            color='blue', s=100, label="Start")
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = sim_predictive.obstacle_pos[0] + sim_predictive.safety_distance * np.cos(u) * np.sin(v)
    y = sim_predictive.obstacle_pos[1] + sim_predictive.safety_distance * np.sin(u) * np.sin(v)
    z = sim_predictive.obstacle_pos[2] + sim_predictive.safety_distance * np.cos(v)
    ax.plot_surface(x, y, z, color='r', alpha=0.2)
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("End-Effector Trajectory Comparison")
    ax.legend()

    sim_predictive.set_axes_equal(ax)
    
    plt.show(block=True)

def main():
    compare_simulations()

if __name__ == "__main__":
    main()
