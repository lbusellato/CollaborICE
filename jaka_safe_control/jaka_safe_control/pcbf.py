import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, fsolve, minimize_scalar
import time
import cvxpy as cp
from quadprog import solve_qp

class PCBFSimulation:
    def __init__(self):
        self.sim_case = 2  # 1 for parallel, 2 for left turn
        self.T = 2.5 # Prediction horizon
        self.dt = 0.01  # Simulation time step
        self.vdes = 12  # Desired velocity
        self.k = 1  # Control gain
        self.rho = 8  # Safety threshold

    def run_simulation(self, control_case=1, verbose=False):
        """
        Run the simulation with specified controller
        control_case: 1 for PCBF, 2 for ECBF, 3 for MPC
        """
        # Simulation parameters
        t_end = 8
        t = np.arange(0, t_end + self.dt, self.dt)
        N = len(t)
        
        # Initial conditions
        x0 = np.array([-37, 10, -40, 10])
        
        # Initialize storage arrays
        x = np.zeros((4, N))
        x[:, 0] = x0
        u = np.zeros((2, N-1))
        H = np.zeros(N-1)
        h = np.zeros(N-1)
        compute_time = np.zeros(N-1)
        d = np.zeros(N-1)
        mus = np.zeros((2, N-1))
        xnom = np.zeros((4, N))
        xnom[:, 0] = x0
        dnom = np.zeros(N-1)
        
        # Setup plots if verbose
        if verbose:
            self.setup_verbose_plots(t, H, u, h)
        
        # Main simulation loop
        for i in range(N-1):

            h[i] = self.h_func(x[:, i])
            
            start_time = time.time()
            
            u[:, i], H[i] = self.calculate_u_pcbf(t[i], x[:, i])
            
            
            x[:, i+1] = self.update_x(t[i], t[i+1], x[:, i], u[:, i])
            
            compute_time[i] = time.time() - start_time
            if verbose:
                self.update_verbose_plots(t[i], x[:, i], h, H, u, i)
            
            if i % 10 == 0 and verbose:
                print(f"Progress: {i}/{N-1}, time: {t[i]:.2f}s")

            pos1 = self.lane1(x[0, i])
            pos2 = self.lane2(x[2, i])
            distance = np.linalg.norm(pos1 - pos2)

            d[i] = distance

            mus[:, i] = self.mu_func(t[i], x[:, i])

            xnom[:, i+1] = self.update_x(t[i], t[i+1], xnom[:, i], mus[:, i])

            pos1 = self.lane1(xnom[0, i])
            pos2 = self.lane2(xnom[2, i])
            distance = np.linalg.norm(pos1 - pos2)

            dnom[i] = distance
        
        print(f"Mean computation time: {np.mean(compute_time):.6f} seconds")
        
        # Plot results
        self.plot_results(t, x, u, h, H, d, dnom, mus)
        
        return t, x, u, h, H
    
    def h_func(self, x):
        """
        Compute the safety function h(x)
        h(x) ≥ 0 implies safety
        """
        pos1 = self.lane1(x[0])
        pos2 = self.lane2(x[2])
        distance = np.linalg.norm(pos1 - pos2)
        return self.rho - distance
    
    def h_func_with_gradient(self, x):
        """
        Compute h(x) and its gradient
        """
        pos1 = self.lane1(x[0])
        pos2 = self.lane2(x[2])
        
        distance = np.linalg.norm(pos1 - pos2)
        h_val = self.rho - distance
        
        # Compute gradient using finite differences
        dx = np.zeros(4)
        delta = 0.01
        for i in range(4):
            x_new1 = x.copy()
            x_new2 = x.copy()
            x_new1[i] += delta
            x_new2[i] -= delta
            h1 = self.rho - np.linalg.norm(self.lane1(x_new1[0]) - self.lane2(x_new1[2]))
            h2 = self.rho - np.linalg.norm(self.lane1(x_new2[0]) - self.lane2(x_new2[2]))
            dx[i] = (h1 - h2) / (2 * delta)
        
        return h_val, dx
    
    def lane1(self, z1):
        """Position of vehicle 1"""
        return np.array([z1, 0]) - np.array([0, 1.5])
    
    def lane2(self, z2):
        """Position of vehicle 2 based on simulation case"""
        if self.sim_case == 1:  # Parallel case
            return np.array([0, z2]) + np.array([1.5, 0])
        else:  # Left turn case
            arc_length = np.pi/2 * 4.5
            if z2 <= -3:
                return np.array([0, z2]) + np.array([1.5, 0])
            elif z2 <= -3 + arc_length:
                theta = (z2 + 3) / arc_length * np.pi/2
                return np.array([-3 + 4.5*np.cos(theta), -3 + 4.5*np.sin(theta)])
            else:
                return np.array([-3 - (z2 + 3 - arc_length), 1.5])
    
    def mu_func(self, t, x):
        """Nominal control law"""
        return self.k * np.array([self.vdes - x[1], self.vdes - x[3]])
    
    def update_x(self, t1, t2, x1, u):
        """Update state from t1 to t2 with control input u"""
        f = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])
        g = np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1]
        ])
        return x1 + (f @ x1 + g @ u) * (t2 - t1)
    
    def path_func(self, tau, t, x):
        """
        Compute the predicted state at time tau given current state x at time t
        """
        delta_t = tau - t
        
        # Predicted positions
        z1 = x[0] + self.vdes * delta_t + (x[1] - self.vdes) / self.k * (1 - np.exp(-self.k * delta_t))
        z2 = x[2] + self.vdes * delta_t + (x[3] - self.vdes) / self.k * (1 - np.exp(-self.k * delta_t))
        
        # Predicted velocities
        z1dot = self.vdes + (x[1] - self.vdes) * np.exp(-self.k * delta_t)
        z2dot = self.vdes + (x[3] - self.vdes) * np.exp(-self.k * delta_t)
        
        predicted_state = np.array([z1, z1dot, z2, z2dot])
        
        # Compute partial derivatives if needed
        if tau == t:
            # Handle the case where tau = t to avoid division by zero
            dtau = np.zeros(4)
            dt = np.zeros(4)
            dx = np.eye(4)
        else:
            # Partial derivatives with respect to tau
            dtau = np.array([
                self.vdes + (x[1] - self.vdes) * np.exp(-self.k * delta_t),
                -self.k * (x[1] - self.vdes) * np.exp(-self.k * delta_t),
                self.vdes + (x[3] - self.vdes) * np.exp(-self.k * delta_t),
                -self.k * (x[3] - self.vdes) * np.exp(-self.k * delta_t)
            ])
            
            # Partial derivatives with respect to t
            dt = -dtau
            
            # Partial derivatives with respect to x
            dx = np.zeros((4, 4))
            dx[0, 0] = 1
            dx[0, 1] = (1 - np.exp(-self.k * delta_t)) / self.k
            dx[1, 1] = np.exp(-self.k * delta_t)
            dx[2, 2] = 1
            dx[2, 3] = (1 - np.exp(-self.k * delta_t)) / self.k
            dx[3, 3] = np.exp(-self.k * delta_t)
        
        return predicted_state, dtau, dt, dx
    
    def find_max(self, t, x):
        """
        Find the time tau that maximizes h(path(tau, t, x))
        """
        def objective(tau):
            predicted_state = self.path_func(tau, t, x)[0]
            return -self.h_func(predicted_state)  # Negative since we're minimizing
        
        # Find tau that maximizes h
        result = minimize_scalar(objective, bounds=(t, t + self.T), method='bounded')
        tau = result.x
        
        h_of_tau = -result.fun  # Convert back to maximum
        
        # Compute derivative of tau with respect to x if needed
        dtau_dx = np.zeros(4)
        delta = 0.01
        for i in range(4):
            x_new = x.copy()
            x_new[i] += delta
            
            # Find new tau with perturbed state
            result_new = minimize_scalar(
                lambda tau: -self.h_func(self.path_func(tau, t, x_new)[0]),
                bounds=(t, t + self.T),
                method='bounded'
            )
            new_tau = result_new.x
            dtau_dx[i] = (new_tau - tau) / delta
            
        return tau, h_of_tau, dtau_dx
    
    def find_zero(self, tau, t, x):
        """
        Find the time eta where h(path(eta, t, x)) = 0
        """
        def objective(eta):
            predicted_state = self.path_func(eta, t, x)[0]
            return self.h_func(predicted_state)
        
        # Try to find the zero between t and tau
        try:
            # Use bisection method to find the zero
            eta = self.bisection_root(objective, t, tau, tol=1e-5)
            
            # If eta > tau or not found, try a different approach
            if eta > tau or np.isnan(eta):
                # Sample points and find the one closest to zero
                z_values = np.linspace(t, tau, 100)
                h_values = np.array([objective(z) for z in z_values])
                closest_idx = np.argmin(np.abs(h_values))
                start_point = z_values[closest_idx]
                
                # Try again with the better starting point
                eta = self.bisection_root(objective, max(t, start_point - 0.1), min(tau, start_point + 0.1), tol=1e-5)
        except:
            # If all else fails, return t
            eta = t
            print(f"Warning: Could not find zero crossing at t = {t}")
        
        # Ensure eta is between t and tau
        if eta < t:
            eta = t
            raise RuntimeError(f"eta < t")
        if eta > tau:
            raise RuntimeError(f"eta > tau")
        
        p, dp_dtau, _, dp_dx = self.path_func(eta, t, x)
        _, dh = self.h_func_with_gradient(p)
        dx = -dh @ dp_dx / (dh @ dp_dtau)
        
        return eta, dx
    
    def bisection_root(self, f, a, b, tol=1e-5, max_iter=100):
        """
        Find root of f in interval [a,b] using bisection method
        """
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
                # Sample intermediate points to look for sign change
                x_vals = np.linspace(a, b, 20)
                f_vals = np.array([f(x) for x in x_vals])
                idx = np.argmin(np.abs(f_vals))
                if abs(f_vals[idx]) < tol:
                    return x_vals[idx]
                
                # Try different intervals
                for i in range(len(x_vals)-1):
                    if f_vals[i] * f_vals[i+1] <= 0:
                        return self.bisection_root(f, x_vals[i], x_vals[i+1], tol, max_iter)
                
                # If we get here, there's no zero crossing
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b
        
        # Apply bisection method
        c = a
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
        Compute the m function and its derivative
        """
        m = 16 / ((self.T) ** 2)
        out = m * (lambda_val ** 2)
        dlambda = 2 * m * lambda_val
        return out, dlambda
    
    def hstar_func(self, t, x):
        """
        Compute the CBF value and its derivatives
        """
        # Get system dynamics
        g = np.array([
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1]
        ])
        
        # Find maximum over prediction horizon
        m1star, h_of_m1star, dm1star_dx = self.find_max(t, x)
        
        # Determine if we need to find zero crossing
        if h_of_m1star <= 0:
            r = m1star
            dr_dx = dm1star_dx
        else:
            r, dr_dx = self.find_zero(m1star, t, x)
            
            # Switch to M1star if derivative gets too large
            if np.linalg.norm(dr_dx) >= 10 * np.linalg.norm(dm1star_dx):
                dr_dx = dm1star_dx
        
        # Compute m function
        m_val, dm_val = self.m_func(r - t)
        
        # Compute Hstar
        hstar = h_of_m1star - m_val
        
        # Compute gradient of h with respect to x at the predicted state
        predicted_state, _, _, dp_of_tau_dx = self.path_func(m1star, t, x)
        _, dh_of_tau_dx = self.h_func_with_gradient(predicted_state)
        
        # Compute derivatives for control computation
        if m1star <= t + 1e-4:
            # Case iii - M1star is at the current time
            delta = 0.01
            path_plus_delta = self.path_func(m1star + delta, t, x)[0]
            h_plus_delta = self.h_func(path_plus_delta)
            dh_dtau = (h_plus_delta - h_of_m1star) / delta
            
            # Since the system is of high relative degree, dtau_dt = 1
            dhstar_dt = dh_dtau * 1
            dhstar_du = dh_of_tau_dx @ dp_of_tau_dx @ g
            
        elif m1star < t + self.T:
            # Case i - M1star is within the prediction horizon
            dhstar_dt = dm_val
            dhstar_du = (dh_of_tau_dx @ dp_of_tau_dx - dm_val * dr_dx) @ g
            
        elif m1star <= t + self.T + 1e-4:
            # Case ii/iii - M1star is at the prediction horizon
            if h_of_m1star > 0:
                # Case ii
                delta = 0.01
                path_plus_delta = self.path_func(m1star + delta, t, x)[0]
                h_plus_delta = self.h_func(path_plus_delta)
                dh_dtau = (h_plus_delta - h_of_m1star) / delta
                
                # Assume dtau_dt = 1 for simplicity
                dhstar_dt = dm_val + dh_dtau * 1
                dhstar_du = (dh_of_tau_dx @ dp_of_tau_dx - dm_val * dr_dx) @ g
            else:
                # Case iii
                delta = 0.01
                path_plus_delta = self.path_func(m1star + delta, t, x)[0]
                h_plus_delta = self.h_func(path_plus_delta)
                dh_dtau = (h_plus_delta - h_of_m1star) / delta
                
                # Assume dtau_dt = 1 for simplicity
                dhstar_dt = dh_dtau * 1
                dhstar_du = dh_of_tau_dx @ dp_of_tau_dx @ g
        else:
            print("Warning: M1star > t+T")
            dhstar_dt = 0
            dhstar_du = np.zeros(2)
            
        return hstar, dhstar_dt, dhstar_du
    
    def calculate_u_pcbf_alt(self, t, x):
        """
        Calculate control input using quadprog.
        """
        # Barrier function calculation
        hstar, dhstar_dt, dhstar_du = self.hstar_func(t, x)
        
        # Baseline control input
        mu = self.mu_func(t, x)
        
        # Simple proportional control with barrier constraint
        alpha = 1
        # Define the quadratic cost matrix and linear term
        J = np.eye(2)         # This is our G matrix in quadprog (2x2 identity)
        F = np.zeros(2)       # This is used to form a = -F (but remains zeros here)
        
        # Barrier function derivative with respect to control input
        A = dhstar_du.reshape(1, 2)   # Shape (1,2)
        b_val = np.array([-alpha * hstar - dhstar_dt])  # Shape (1,)
        
        # Set up quadprog parameters:
        # The problem is: minimize 0.5*du.T*J*du + F.T*du.
        # quadprog solves: minimize 0.5*du.T*G*du - a.T*du.
        G = J
        a = -F  # since F is zero, a is also a zero vector
        
        # Convert A*du <= b to quadprog form:
        # Multiply by -1: -A*du >= -b.
        # quadprog expects C^T*du >= d.
        C = -A.T     # C has shape (2,1)
        d = -b_val   # d has shape (1,)
        
        # Solve QP using quadprog
        try:
            sol = solve_qp(G, a, C, d, meq=0)
            du = sol[0]  # The optimal adjustment vector
        except Exception as e:
            raise RuntimeError("QP solve failed") from e
        
        if np.any(np.isnan(du)):
            raise RuntimeError("QP solve returned NaN values")
        
        # Compute the final control input
        u = mu + du
        
        return u, hstar
        
    def calculate_u_pcbf(self, t, x):
        """
        Calculate control input
        """
        # Barrier function calculation
        hstar, dhstar_dt, dhstar_du = self.hstar_func(t, x)
        
        # Baseline control input
        mu = self.mu_func(t, x)
        
        # Simple proportional control with barrier constraint
        alpha = 1
        J = np.eye(2)
        F = np.zeros(2)
        A = dhstar_du.reshape(1, 2)
        b = np.array([-alpha * hstar - dhstar_dt])
        
        du = cp.Variable(2)
        objective = cp.Minimize(0.5 * cp.quad_form(du, J) + F.T @ du)
        constraints = [A @ du <= b]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        if du.value is None or np.any(np.isnan(du.value)):
            raise RuntimeError('QP solve failed')
        
        u = mu + du.value

        return u, hstar
    
    def setup_verbose_plots(self, t, H, u, h):
        """Setup plots for verbose mode"""
        plt.figure(8)
        self.H_plot, = plt.plot(t[:-1], H)
        plt.xlabel('Time (s)')
        plt.ylabel('H*')
        plt.axis([0, 7, -55, 10])
        plt.title('H* vs Time')
        
        plt.figure(9)
        self.u1_plot, = plt.plot(t[:-1], u[0, :])
        self.u2_plot, = plt.plot(t[:-1], u[1, :])
        plt.xlabel('Time (s)')
        plt.ylabel('u')
        plt.legend(['u_1', 'u_2'])
        plt.axis([0, 7, -10, 10])
        plt.title('Control Inputs vs Time')
        
        plt.figure(10)
        self.h_plot, = plt.plot(t[:-1], h)
        plt.plot([0, 8], [0, 0], 'r--')
        plt.xlabel('Time (s)')
        plt.ylabel('Predicted Safety')
        plt.axis([0, 8, -60, 10])
        plt.title('Safety Constraint vs Time')
        
        plt.ion()
        plt.show()
    
    def update_verbose_plots(self, t, x, h, H, u, i):
        """Update plots for verbose mode"""
        # Predict future trajectory
        t_pred = np.arange(t, 8 + 0.1, 0.1)
        h_pred = np.zeros_like(t_pred)
        
        for j, tj in enumerate(t_pred):
            pred_state = self.path_func(tj, t, x)[0]
            h_pred[j] = self.h_func(pred_state)
        
        plt.figure(10)
        self.h_plot.set_xdata(t_pred)
        self.h_plot.set_ydata(h_pred)
        
        plt.figure(9)
        self.u1_plot.set_ydata(u[0, :i+1])
        self.u2_plot.set_ydata(u[1, :i+1])
        
        plt.figure(8)
        self.H_plot.set_ydata(H[:i+1])
        
        plt.pause(0.001)
    
    def plot_results(self, t, x, u, h, H, d, dnom, mus):
        """Plot simulation results"""
        # Plot vehicle positions
        plt.figure(1)
        # Animate positions
        l1 = self.lane1(x[0, 0])
        l2 = self.lane2(x[2, 0])
        p1 = plt.plot(l1[0], l1[1], 'bo', markerfacecolor='b')[0]
        p2 = plt.plot(l2[0], l2[1], 'go', markerfacecolor='g')[0]
        
        # Plot lanes
        plt.plot([-50, -3], [3, 3], 'k')
        plt.plot([-50, -3], [-3, -3], 'k')
        plt.plot([50, 3], [3, 3], 'k')
        plt.plot([50, 3], [-3, -3], 'k')
        plt.plot([-3, -3], [-50, -3], 'k')
        plt.plot([3, 3], [-50, -3], 'k')
        plt.plot([-3, -3], [50, 3], 'k')
        plt.plot([3, 3], [50, 3], 'k')
        
        plt.axis('equal')
        plt.axis([-50, 50, -50, 50])
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')
        plt.title('Vehicle Trajectories')
        
        # Animate
        for i in range(0, len(t), 10):
            l1 = self.lane1(x[0, i])
            l2 = self.lane2(x[2, i])
            p1.set_xdata(l1[0])
            p1.set_ydata(l1[1])
            p2.set_xdata(l2[0])
            p2.set_ydata(l2[1])
            plt.pause(0.01)
        
        # Plot control inputs
        plt.figure(2)
        plt.plot(t[:-1], u.T)
        plt.plot(t[:-1], d)
        plt.plot(t[:-1], dnom)
        plt.plot(t[:-1], mus.T)
        plt.xlabel('Time (s)')
        plt.ylabel('u (m/s^2)')
        plt.legend(['u_1', 'u_2', 'd', 'dnom', 'unom_1', 'unom_2'])
        plt.title('Control Inputs')
        plt.grid()
        
        # Plot positions
        plt.figure(3)
        plt.plot(t, x[[0, 2], :].T)
        plt.xlabel('Time (s)')
        plt.ylabel('z_i (meters)')
        plt.legend(['z_1', 'z_2'])
        plt.title('Vehicle Positions')
        
        # Plot velocities
        plt.figure(4)
        plt.plot(t, x[[1, 3], :].T)
        plt.plot([t[0], t[-1]], [12, 12], 'k--')
        plt.xlabel('Time (s)')
        plt.ylabel('z_i_dot (m/s)')
        plt.legend(['z_1_dot', 'z_2_dot'])
        plt.title('Vehicle Velocities')
        
        # Plot constraints
        plt.figure(5)
        plt.plot(t[:-1], H)
        plt.plot(t[:-1], h)
        plt.xlabel('Time (s)')
        plt.ylabel('Constraints')
        plt.legend(['H*', 'h'])
        plt.title('Safety Constraints')
        
        plt.show()


def main():
    sim = PCBFSimulation()
    
    sim.run_simulation(control_case=1, verbose=False)


if __name__ == "__main__":
    main()