import rclpy
import rclpy.logging
import numpy as np
import cvxpy as cp
import scipy.spatial.transform as transform
import json
import threading
import time

from std_msgs.msg import String
from jaka_messages.srv import TargetReached
from jaka_interface.interface import JakaInterface
from jaka_interface.data_types import MoveMode
from jaka_interface.pose_conversions import leap_to_jaka
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R, Slerp
from tf_transformations import euler_from_quaternion, quaternion_from_euler

class JAKA(Node):
    def __init__(self):
        # Init node
        super().__init__('vanilla_cbf_node')

        self.declare_parameter('publish_robot_state', rclpy.Parameter.Type.BOOL)
        self.publish_robot_state = self.get_parameter('publish_robot_state').value
        
        # Init the robot interface
        self.jaka_interface = JakaInterface(publish_state=self.publish_robot_state)
        self.jaka_interface.initialize()

        # Logging
        self.logger = rclpy.logging.get_logger('vanilla_cbf')

        self.t = 0
        self.dt = 0.008
        self.control_loop_timer = self.create_timer(self.dt, self.control_loop) 

        self.min_hand_confidence = 0.33
        self.hand_radius = 0
        self.max_joint_velocity = 100 #0.95 * np.pi
        self.qd_prev = np.zeros(6)

        # Obstacle position
        self.obstacle_pos_sub = self.create_subscription(String, '/sensors/leap/json', self.obstacle_callback, 1)
        self.obstacle_pos = None

        self.path_function = self.triangle_wave

        # Go to the starting position
        self.jaka_interface.robot.disable_servo_mode()
        self.jaka_interface.robot.linear_move([-400, 300, 300, -np.pi, 0, -20*np.pi/180], MoveMode.ABSOLUTE, 100, True)
        self.jaka_interface.robot.enable_servo_mode()


    def control_loop(self):
        q = np.array(self.jaka_interface.robot.joint_position)
        tcp = np.array(self.jaka_interface.robot.tcp_position)
        q_des = np.array(self.path_function(self.t))

        qd_des = (q_des - q) / self.dt
        
        if self.obstacle_pos is not None:
            qd_safe = self.safe_controller(qd_des, tcp[:3], self.obstacle_pos)
        else:
            qd_safe = qd_des
        
        q_out = np.array(self.jaka_interface.robot.joint_position) + qd_safe * self.dt

        self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)
        self.t += self.dt
    
    #########################################
    #                                       #
    # Control Barrier Functions             #
    #                                       #
    #########################################

    def safe_controller(self, qd_des, tcp, obstacle, min_distance: int=100):

        qd_opt = cp.Variable(len(qd_des))

        # Define the cost function (same as MATLAB)
        H = np.eye(len(qd_des))  # Identity matrix
        f = -qd_des  # Linear term

        # Quadratic cost function
        objective = cp.Minimize((1/2) * cp.quad_form(qd_opt, H) + f.T @ qd_opt)

        J = self.jaka_interface.robot.jacobian()

        d = np.linalg.norm(tcp - obstacle)
        
        #if d < 0.9 * min_distance:
        #    self.loginfo('Possible collision detected!')
        
        gamma = 0.8  
        h = d - self.hand_radius - min_distance
        n = (tcp - obstacle) / d  # Normal vector
        grad_h = n @ J[:3, :]  # Compute gradient
        constraints = [
            -grad_h.reshape(1, -1) @ qd_opt <= gamma * h,
            qd_opt >= -self.max_joint_velocity,
            qd_opt <= self.max_joint_velocity]

        # Solve the QP problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)

        # Get the optimized joint velocity
        if qd_opt.value is None:
            raise RuntimeError('solve QP failed')
        qd_optimal = qd_opt.value

        alpha = 0.2  # Smoothing factor (0 < alpha <= 1), adjust as needed
        qd_smoothed = alpha * qd_optimal + (1 - alpha) * self.qd_prev
        self.qd_prev = qd_smoothed 

        return qd_smoothed
    
    #########################################
    #                                       #
    # Path Functions                        #
    #                                       #
    #########################################

    def linear(self, current_pos, target_tcp, max_step_speed):
        
        target_pos = np.array(target_tcp)
        
        delta = target_pos - current_pos
        distance = np.linalg.norm(delta)
        
        if distance <= max_step_speed:
            next_pos = target_pos
        else:
            next_pos = current_pos + (delta / distance) * max_step_speed
        
        tcp_pose = list(next_pos / 1000) + [np.pi, 0.0, -20*np.pi/180]
        
        next_joint_positions = self.jaka_interface.robot.kine_inverse(None, tcp_pose)
        
        return next_joint_positions

    def triangle_wave(self, tau: float, 
                      t0: float=0.0, 
                      center: list=[-400.0, 0.0, 300.0], 
                      orientation: list=[np.pi, 0.0, -20*np.pi/180],
                      amplitude: float=300.0, 
                      frequency: float=0.1)->tuple:
        """Computes the nominal end-effector trajectory, both in Cartesian and joint space, for a fixed orientation triangle
        wave on the y direction.

        Parameters
        ----------
        tau : float
            Time to compute the trajectory in.
        t0 : float, optional
            Start time of the trajectory, by default 0.0
        center : list, optional
            The coordinates in MILLIMETERS of the triangle wave's zero, by default [-400.0, 0.0, 300.0]
        orientation : list, optional
            The fixed orientation for the TCP, by default [np.pi, 0.0, 0.0]
        amplitude : float, optional
            The amplitude in MILLIMETERS of the triangle wave, by default 100.0
        frequency : float, optional
            The frequency in Hertz of the triangle wave, by default 0.1

        Returns
        -------
        tuple
            The computed TCP pose and joint positions at time tau.
        """
        x_c, y_c, z_c = center

        period = 1.0 / frequency
        phase = ((tau - t0) % period) / period 
        
        x = x_c
        y = y_c + amplitude * (4 * np.abs(phase - 0.5) - 1) 
        z = z_c

        tcp_pose = np.array([x/1000, y/1000, z/1000, *orientation])

        joint_pose = self.jaka_interface.robot.kine_inverse(None, tcp_pose)

        return joint_pose
    
    #########################################
    #                                       #
    # Subscriber callbacks                  #
    #                                       #
    #########################################
    
    def obstacle_callback(self, msg: String):
        data = json.loads(msg.data)
        hands = data.get('hands')
        if hands:
            confidences = np.array([h.get('confidence') for h in hands])
            hand = hands[np.argmax(confidences)]
            if hand.get('confidence') < self.min_hand_confidence:
                self.obstacle_pos = None
            else:
                hand = hands[0]
                keypoints = hand.get('hand_keypoints')

                fingers = keypoints.get("fingers", {})
                palm_position = np.array(keypoints.get('palm_position'))

                max_dist = 0     
                # TODO: probably reduce the search space to the joints we KNOW are far away      
                for _, joints in fingers.items():
                    for _, pos in joints.items():
                        dist = np.linalg.norm(np.abs(np.array(pos['prev_joint']) - palm_position))
                        if dist > max_dist:
                            max_dist = dist
                self.hand_radius = max_dist * 1000

                # Convert to JAKA world
                palm_position = leap_to_jaka(palm_position)
                palm_position = np.array(palm_position) * 1000 # to mm
                self.obstacle_pos = palm_position
        else:
            self.obstacle_pos = None


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