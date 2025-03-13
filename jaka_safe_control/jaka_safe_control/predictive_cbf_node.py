import rclpy
import rclpy.logging
import threading
import numpy as np
import time

from jaka_interface.JakaInterface import JakaInterface
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from typing import List, Tuple
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose

class JAKA(Node):
    def __init__(self):
        # Init node
        super().__init__('predictive_cbf_node')

        self.declare_parameter('publish_robot_state', rclpy.Parameter.Type.BOOL)
        self.publish_robot_state = self.get_parameter('publish_robot_state').value
        
        # Init the robot interface
        self.jaka_interface = JakaInterface(publish_state=self.publish_robot_state)
        self.jaka_interface.initialize()

        # Logging
        self.logger = rclpy.logging.get_logger('predictive_cbf')

        # The control loop will attempt to maintain a 125Hz frequency, matching the controller's
        self.t = 0.0
        self.dt = 0.008
        self.control_loop_timer = self.create_timer(self.dt, self.control_loop) 

        # PCBF parameters
        self.horizon = 1 # Look ahead 1 second
        # Don't need too much granularity here, let's align to the forecasting's
        #self.steps = int(self.horizon / self.dt)
        self.steps = 30 
        self.safe_distance = 100 # mm

        # Hand tracking
        self.hand_position = np.array([-400, 0, 300])

        # Go to the starting position
        self.jaka_interface.disable_servo_mode()
        self.jaka_interface.moveL([-400, 300, 300, -np.pi, 0, 0], JakaInterface.ABS, True, 200)
        self.jaka_interface.enable_servo_mode()
        
        self.poseA = Pose()
        self.poseA.position.x = -400.0
        self.poseA.position.y = -300.0
        self.poseA.position.z = 300.0
        self.poseA.orientation.x = 1.0
        self.poseA.orientation.y = 0.0
        self.poseA.orientation.z = 0.0
        self.poseA.orientation.w = 0.0
        self.jaka_interface.inverse_kinematics(self.poseA)

        self.tcp = self.jaka_interface.get_tcp_position(type=np.ndarray)

        self.cnt = 0

    def control_loop(self):
        if self.cnt < 2:
            start = time.time()
            tcp = self.path_function(self.t)
            self.jaka_interface.servoP(tcp, JakaInterface.ABS, step_num=1)
            end = time.time()
            self.dt = end - start
            self.t += self.dt
            self.cnt += 1
            self.loginfo(f"{(self.jaka_interface.get_tcp_position(type=np.ndarray)[1]-tcp[1])}    dt:{self.dt}")
       
    #########################################
    #                                       #
    # Predictive Control Barrier Function   #
    #                                       #
    #########################################

    def find_maximizers(self):      
        # Propagate the trajectory forward  
        future_positions = self.propagate_path_function(self.t, self.horizon, self.steps)

        future_h_values = []

        for i, ee_pos in enumerate(future_positions):
            h_tau = self.constraint_function(ee_pos, self.safe_distance)
            tau = np.round(self.t + i*self.dt, 3)
            future_h_values.append((tau, h_tau))
        local_maxima = []
        
        root = None
        for i in range(1, len(future_h_values) - 1):
            tau_prev, h_prev = future_h_values[i - 1]
            if abs(h_prev) < 1:
                root = tau_prev
            curr_tau, h_curr = future_h_values[i]
            _, h_next = future_h_values[i + 1]
            if h_prev < h_curr > h_next: 
                local_maxima.append((root, curr_tau, h_curr))

        if not local_maxima:
            return []

        # Always keep the first max
        M_set = [local_maxima[0]]  

        for i in range(1, len(local_maxima)):
            root, tau_i, h_i = local_maxima[i]
            tau_prev, h_prev = M_set[-1]

            # If the current max is part of a new peak, keep it
            if tau_i > tau_prev + self.dt:
                M_set.append((root, tau_i, h_i))

        return M_set

    def constraint_function(self, tcp_position: np.ndarray, d_safe: float=100)->float:
        """Compute the constraint function:\n
        h(tau, p(tau; t,x)) = d_safe - d(tcp_position - p(tau; t,x))

        Parameters
        ----------
        tcp_position : np.ndarray
            The position of the TCP.
        d_safe : float, optional
            The minimum distance in MILLIMETERS, by default 100

        Returns
        -------
        float
            The computed constraint function value.
        """
        d_actual = int(np.linalg.norm(tcp_position[:3] - self.hand_position)) # TODO: hand_position becomes argument when using leap
        
        h = d_safe - d_actual

        return h

    def propagate_path_function(self, t_start: float, horizon: float, steps: int = 10)->List[np.ndarray]:
        """Propagate the nominal trajectory from t_start to a given horizon, generating a fixed number of intermediate 
        steps.

        Parameters
        ----------
        t_start : float
            The current time.
        horizon : float
            The time horizon.
        steps : int, optional
            The number of points to generate, by default 10

        Returns
        -------
        List[np.ndarray]
            The propagated trajectory.
        """
        dt = horizon / steps
        future_positions = []
        
        for i in range(steps):
            t_future = t_start + i * dt
            next_pose, _ = self.path_function(t_future)
            future_positions.append(next_pose)

        return future_positions

    def path_function(self, tau: float, 
                      t0: float=0.0, 
                      center: List[float]=[-400.0, 0.0, 300.0], 
                      orientation: List[float]=[np.pi, 0.0, 0.0],
                      amplitude: float=300.0, 
                      frequency: float=0.1)->Tuple[Pose, JointState]:
        """Computes the nominal end-effector trajectory, both in Cartesian and joint space, for a fixed orientation triangle
        wave on the y direction.

        Parameters
        ----------
        tau : float
            Time to compute the trajectory in.
        t0 : float, optional
            Start time of the trajectory, by default 0.0
        center : List[float], optional
            The coordinates in MILLIMETERS of the triangle wave's zero, by default [-400.0, 0.0, 300.0]
        orientation : List[float], optional
            The fixed orientation for the TCP, by default [np.pi, 0.0, 0.0]
        amplitude : float, optional
            The amplitude in MILLIMETERS of the triangle wave, by default 100.0
        frequency : float, optional
            The frequency in Hertz of the triangle wave, by default 0.1

        Returns
        -------
        Tuple[Pose, JointState]
            The computed TCP pose and joint positions at time tau.
        """
        x_c, y_c, z_c = center

        period = 1.0 / frequency
        phase = ((tau - t0) % period) / period 
        
        x = x_c
        y = y_c + amplitude * (4 * np.abs(phase - 0.5) - 1) 
        z = z_c

        tcp_pose = np.array([x, y, z, *orientation])

        return tcp_pose
    
       
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