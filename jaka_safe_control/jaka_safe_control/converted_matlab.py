import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import cvxpy as cp

class VehicleControlSimulation:
    def __init__(self, sim_case=1):
        self.sim_case = sim_case
        
        # Simulation parameters
        self.dt = 0.01
        self.t_end = 8
        self.t = np.arange(0, self.t_end, self.dt)
        
        # Initial conditions
        self.x0 = np.array([-37, 10, -40, 10])
        
        # Desired velocity
        self.vdes = 12
        self.k = 1  # Control gain
        
        # Prediction horizon
        self.T = 2.5
        
        # Initialize state and control storage
        self.N = len(self.t)
        self.x = np.zeros((4, self.N))
        self.u = np.zeros((2, self.N-1))
        self.h = np.zeros(self.N-1)
        self.H = np.zeros(self.N-1)

        # Dynamics
        self.f = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        self.g = np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1]
        ])
        
        self.x[:, 0] = self.x0

    def lane1(self, z1):
        """
        First lane point calculation
        """
        return np.array([z1, 0]) - np.array([0, 1.5])

    def lane2(self, z2):
        """
        Second lane point calculation (depends on simulation case)
        """
        arc_length = np.pi/2 * 4.5
        
        if self.sim_case == 1:
            # Parallel lanes
            return np.array([0, z2]) + np.array([1.5, 0])
        elif self.sim_case == 2:
            # Left turn scenario
            if z2 <= -3:
                return np.array([0, z2]) + np.array([1.5, 0])
            elif z2 <= -3 + arc_length:
                theta = (z2 + 3) / arc_length * np.pi/2
                return np.array([-3 + 4.5 * np.cos(theta), -3 + 4.5 * np.sin(theta)])
            else:
                return np.array([-3 - (z2 + 3 - arc_length), 1.5])

    def h_func_real(self, x):
        """
        Safety constraint function
        """
        rho = 2
        return rho - np.linalg.norm(self.lane1(x[0]) - self.lane2(x[2]))

    def h_func(self, t, x):
        out = self.h_func_real(x)
        dx = np.zeros(4)
        delta = 0.01
        for i in range(4):
            x_new1 = x
            x_new2 = x
            x_new1[i] = x_new1[i] + delta
            x_new2[i] = x_new2[i] - delta
            dx[i] = (self.h_func_real(x_new1) - self.h_func_real(x_new2)) / (2 * delta)
        return out, dx

    def mu_func(self, x):
        """
        Baseline control input to track desired velocity
        """
        return self.k * np.array([self.vdes - x[1], self.vdes - x[3]])

    def path_func(self, tau, t, x):
        """
        Predict future trajectory
        """
        delta_t = tau - t
        
        def z_future(delta_t, z, zdot):
            return z + self.vdes * delta_t + (zdot - self.vdes) / self.k * (1 - np.exp(-self.k * delta_t))
        
        def zdot_future(delta_t, zdot):
            return self.vdes + (zdot - self.vdes) * np.exp(-self.k * delta_t)
        
        p = np.array([
            z_future(delta_t, x[0], x[1]),
            zdot_future(delta_t, x[1]),
            z_future(delta_t, x[2], x[3]),
            zdot_future(delta_t, x[3])
        ])

        def z_future_dtau(delta_t, zdot):
            return self.vdes + (zdot - self.vdes) * np.exp(-self.k * delta_t)

        def zdot_future_dtau(delta_t, zdot):
            return -self.k * zdot * np.exp(-self.k * delta_t)

        def z_future_dx(delta_t):
            return np.array([1, (1 - np.exp(-self.k * delta_t)) / self.k])

        def zdot_future_dx(delta_t):
            return np.array([0, np.exp(-self.k * delta_t)])
        
        dtau = np.array([z_future_dtau(tau-t,x[1]),
                zdot_future_dtau(tau-t,x[1]),
                z_future_dtau(tau-t,x[3]),
                zdot_future_dtau(tau-t,x[3])])

        dt = -dtau

        dx = np.array([[*z_future_dx(tau-t),    0, 0],
            [*zdot_future_dx(tau-t), 0, 0],
            [0, 0,                  *z_future_dx(tau-t)],
            [0, 0,                  *zdot_future_dx(tau-t)]])

        return p, dtau, dt, dx

    def _find_max(self, t, x):
        def h_at_tau(tau):
            pred_x, _, _, _ = self.path_func(tau, t, x)
            h_val, _ = self.h_func(t, pred_x)
            return h_val
        n_samples = 100
        tau_samples = np.linspace(t, t + self.T, n_samples)
        h_values = [h_at_tau(tau) for tau in tau_samples]

        first_max_index = None
        for i in range(1, len(h_values) - 1):
            if h_values[i] > h_values[i - 1] and h_values[i] >= h_values[i + 1]:
                first_max_index = i
                break

        if first_max_index is None:
            first_max_index = np.argmax(h_values)
        
        tau_candidate = tau_samples[first_max_index]

        dtau = tau_samples[1] - tau_samples[0]
        lower_bound = max(t, tau_candidate - dtau)
        upper_bound = min(t + self.T, tau_candidate + dtau)

        def neg_h(tau):
            return -h_at_tau(tau)

        result = optimize.minimize_scalar(neg_h, bounds=(lower_bound, upper_bound), method='bounded')
        tau_opt = result.x
        h_opt = h_at_tau(tau_opt)

        return tau_opt, h_opt        

    def find_max(self, t, x):
        tau, h_of_tau = self._find_max(t, x)

        dx = np.zeros(4)
        delta = 0.01
        for i in range(4):
            x_new = x
            x_new[i] = x_new[i] + delta
            new_tau, _ = self._find_max(t, x_new)
            dx[i] = (new_tau - tau) / delta

        return tau, h_of_tau, dx     

    def my_fzero(self, func, t1, t2):
        h1, _ = func(t1)
        h2, _ = func(t2)
        
        if h1 > 0:
            return t1
        elif h2 < 0:
            raise ValueError("Invalid bounds.")
        
        tau = (t1 + t2) / 2.0
        h, _ = func(tau)
        count = 0
        while abs(h) > 1e-5:
            count += 1
            if h < 0:
                t1 = tau
            else:
                t2 = tau
            
            tau = (t1 + t2) / 2.0
            h = func(tau)
            
            if count > 1000:
                print("Failure in my_fzero.")
                break
            
        return tau

    def find_zero(self, tau, t, x):
        func = lambda z: self.h_func(z, self.path_func(z, t, x))
        
        # Find eta such that func(eta) is approximately zero.
        eta = self.my_fzero(func, t, tau)
        
        # Compute derivatives:
        p, dp_dtau, _, dp_dx = self.path_func(eta, t, x)
        _, dh = self.h_func(eta, p)
        
        # Compute dx using the provided formula.
        dx = -dh * dp_dx / (dh * dp_dtau)
        
        return eta, dx

    def m_func(self, lambda_val):
        """
        Auxiliary function for constraint calculation
        """
        m = 16 / (self.T**2)
        out = m * (lambda_val**2)
        dlambda = 2 * m * lambda_val
        return out, dlambda

    def Hstar_func(self, t, x):
        """
        H* calculation for predictive control barrier function
        """
        # Find maximum time
        M1star, h_of_M1star, dM1star_dx = self.find_max(t, x)

        if h_of_M1star <= 0:
            R = M1star
            dR_dx = dM1star_dx
        else:
            R, dR_dx = self.find_zero(M1star, t, x)
            if np.linalg.norm(dR_dx) >= 10 * np.linalg.norm(dM1star_dx):
                dR_dx = dM1star_dx
            
        m, dm = self.m_func(R - t)
        out = h_of_M1star - m

        p, _, _, dp_of_tau_dx = self.path_func(M1star, t, x)
        _, dh_of_tau_dx = self.h_func(M1star, p)

        if M1star <= t + 1e-4:
            delta = 0.01
            p, _, _, _ = self.path_func(M1star + delta, t, x)
            h, _ = self.h_func(M1star + delta, p)
            dh_dtau = (h - h_of_M1star) / delta
            dt = dh_dtau * 1 # dh_dtau is always 1 for high degree systems
            du = dh_of_tau_dx * dp_of_tau_dx @ self.g
        elif M1star < t + self.T:
            dt = dm
            du = (dh_of_tau_dx * dp_of_tau_dx - dm * dR_dx) @ self.g
        elif M1star <= t + self.T + 1e-4:
            if h_of_M1star > 0:
                delta = 0.01
                p, _, _, _ = self.path_func(M1star + delta, t, x)
                h, _ = self.h_func(M1star + delta, p)
                dh_dtau = (h - h_of_M1star) / delta
                dt = dm + dh_dtau * 1 # assume dh_dtau = 1
                du = (dh_of_tau_dx * dp_of_tau_dx - dm * dR_dx) @ self.g
            else:
                delta = 0.01
                p, _, _, _ = self.path_func(M1star + delta, t, x)
                h, _ = self.h_func(M1star + delta, p)
                dh_dtau = (h - h_of_M1star) / delta
                dt = dh_dtau * 1 # assume dh_dtau = 1
                du = dh_of_tau_dx * dp_of_tau_dx @ self.g
        else:
            raise Exception('Something went wrong, M1star > t + T')

        return out, dt, du

    def update_x(self, t1, t2, x1, u):
        """
        Update vehicle state using simple Euler integration
        """
        
        dx = self.f @ x1 + self.g @ u
        return x1 + dx * (t2 - t1)

    def calculate_u(self, t, x):
        """
        Calculate control input
        """
        # Barrier function calculation
        H , dHdt, dHdu = self.Hstar_func(t, x)
        
        # Baseline control input
        mu = self.mu_func(x)
        
        # Simple proportional control with barrier constraint
        alpha = 1
        J = np.eye(2)
        F = np.zeros(2)
        A = dHdu
        b = -alpha * H - dHdt

        du = cp.Variable(2)
        objective = cp.Minimize(0.5 * cp.quad_form(du, J) + F.T @ du)
        constraints = [A @ du <= b]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if du.value is None or np.any(np.isnan(du.value)):
            raise RuntimeError('QP solve failed')
        
        u = mu + du.value
        
        return u, H

    def run_simulation(self):
        """
        Run the vehicle control simulation
        """
        for i in range(self.N - 1):
            # Calculate safety constraint
            self.h[i], _ = self.h_func(0, self.x[:, i])
            
            # Calculate control input
            u, H = self.calculate_u(self.t[i], self.x[:, i])
            
            # Store control input and barrier function
            self.u[:, i] = u
            self.H[i] = H
            
            # Update vehicle state
            self.x[:, i+1] = self.update_x(
                self.t[i], 
                self.t[i+1], 
                self.x[:, i], 
                u
            )

    def plot_results(self):
        """
        Generate plots of simulation results
        """
        plt.figure(figsize=(15, 10))
        
        # Control Inputs
        plt.subplot(2, 2, 1)
        plt.plot(self.t[:-1], self.u[0, :], label='u1')
        plt.plot(self.t[:-1], self.u[1, :], label='u2')
        plt.title('Control Inputs')
        plt.xlabel('Time (s)')
        plt.ylabel('u (m/s²)')
        plt.legend()
        
        # Vehicle Positions
        plt.subplot(2, 2, 2)
        plt.plot(self.t, self.x[0, :], label='z1')
        plt.plot(self.t, self.x[2, :], label='z2')
        plt.title('Vehicle Positions')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (m)')
        plt.legend()
        
        # Vehicle Velocities
        plt.subplot(2, 2, 3)
        plt.plot(self.t, self.x[1, :], label='zdot1')
        plt.plot(self.t, self.x[3, :], label='zdot2')
        plt.title('Vehicle Velocities')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.legend()
        
        # Constraints
        plt.subplot(2, 2, 4)
        plt.plot(self.t[:-1], self.h, label='h')
        plt.plot(self.t[:-1], self.H, label='H')
        plt.title('Safety Constraints')
        plt.xlabel('Time (s)')
        plt.ylabel('Constraint Value')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

# Run simulation for parallel lanes
sim_parallel = VehicleControlSimulation(sim_case=1)
sim_parallel.run_simulation()
sim_parallel.plot_results()