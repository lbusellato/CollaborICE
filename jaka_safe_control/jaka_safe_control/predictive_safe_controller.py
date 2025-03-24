import cvxpy as cp
import numpy as np

from jaka_interface.real_robot import RealRobot

class PredictiveSafeController():
    def __init__(self,
                 robot_instance: RealRobot,
                 cbf_gamma: float = 0.8,
                 min_distance: float = 100.0,
                 max_joint_velocity: float = 100,
                 lp_filter_alpha: float = 0.2,
                 dt: float = 0.008):
        self.robot = robot_instance
        self.gamma = cbf_gamma
        self.min_dist = min_distance
        self.max_joint_vel = max_joint_velocity
        self.alpha = lp_filter_alpha
        self.dt = dt

        self.qd_prev = np.zeros(6)

        self.qd_opt = cp.Variable(6)
        self.H = np.eye(6)

    def update(self, q_des: np.ndarray, obs_pos: np.ndarray, obs_radius: float)->np.ndarray:
        q = np.array(self.robot.joint_position)
        tcp = np.array(self.robot.tcp_position)[:3]

        qd_out = (q_des - q) / self.dt

        if obs_pos is not None:
            qd_out = self.apply_cbf(qd_out, tcp, obs_pos, obs_radius)

        qd_out = self.apply_lp_filter(qd_out)    

        return q + qd_out * self.dt

    def apply_cbf(self, qd_des: np.ndarray, tcp: np.ndarray, obs_pos: np.ndarray, obs_radius: float)->np.ndarray:
        # Linear term of the cost function
        f = -qd_des
        # Quadratic cost function
        objective = cp.Minimize((1/2) * cp.quad_form(self.qd_opt, self.H) + f.T @ self.qd_opt)

        # Control Barrier Function h
        distance = np.linalg.norm(tcp - obs_pos)
        h = distance - (obs_radius + self.min_dist)

        # Lie derivative of h
        n = (tcp - obs_pos) / distance
        J = self.robot.jacobian()
        grad_h = n @ J[:3, :]

        # Constraints to the QP
        constraints = [
            -grad_h.reshape(1, -1) @ self.qd_opt <= self.gamma * h,
            self.qd_opt >= -self.max_joint_vel,
            self.qd_opt <= self.max_joint_vel
            ]
        
        # Set up and solve the QP
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        if self.qd_opt.value is None:
            raise RuntimeError('solve QP failed')

        return self.qd_opt.value

    
    def apply_lp_filter(self, qd: np.ndarray)->np.ndarray:
        qd = self.alpha * qd + (1 - self.alpha) * self.qd_prev
        self.qd_prev = qd
        return qd