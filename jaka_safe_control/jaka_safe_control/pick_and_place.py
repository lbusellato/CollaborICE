import rclpy
import rclpy.logging
import numpy as np
import cvxpy as cp
import json
import threading
import time

from std_msgs.msg import String
from jaka_interface.interface import JakaInterface
from jaka_interface.data_types import MoveMode
from jaka_interface.pose_conversions import leap_to_jaka
from jaka_safe_control.path_functions import linear_move 
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from scipy.spatial.transform import Rotation as R, Slerp
from tf_transformations import quaternion_from_euler

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
        self.obstacle_pos_sub = self.create_subscription(String, '/sensors/leapScreen/json', self.obstacle_callback, 1)
        self.obstacle_pos = None

        #Baia:                       Pallet:
        #                            /----------------
        #                            |               |
        #                            |        P1     |
        #    ----                    |               |
        #    |B1|                    |  P2           |
        #    |B2|                    |        P3     |
        #    |B3|                    |               |
        #    ----                    -----------------
        # Pick locations
        self.P1_approach = np.array([-150, -467,  25, -np.pi, 0, -20*np.pi/180])
        self.P1_pick     = np.array([-150, -467, -25, -np.pi, 0, -20*np.pi/180])
        self.P2_approach = np.array([-212, -407,  25, -np.pi, 0, -20*np.pi/180])
        self.P2_pick     = np.array([-212, -407, -25, -np.pi, 0, -20*np.pi/180])
        self.P3_approach = np.array([-273, -467,  25, -np.pi, 0, -20*np.pi/180])
        self.P3_pick     = np.array([-273, -467, -25, -np.pi, 0, -20*np.pi/180])
        # Place locations
        self.B1_approach = np.array([-180,  261, 100, -np.pi, 0, -20*np.pi/180])
        self.B1_place    = np.array([-180,  261,  70, -np.pi, 0, -20*np.pi/180])
        self.B2_approach = np.array([-285,  261, 100, -np.pi, 0, -20*np.pi/180])
        self.B2_place    = np.array([-285,  261,  70, -np.pi, 0, -20*np.pi/180])
        self.B3_approach = np.array([-390,  261, 100, -np.pi, 0, -20*np.pi/180])
        self.B3_place    = np.array([-390,  261,  70, -np.pi, 0, -20*np.pi/180])
        
        # Go to the starting position
        self.home = np.array([-212, -467, 300, -np.pi, 0, -20*np.pi/180])
        self.jaka_interface.robot.disable_servo_mode()
        self.jaka_interface.robot.linear_move(self.home, MoveMode.ABSOLUTE, 200, True)
        self.jaka_interface.robot.enable_servo_mode()

        # Pick and place variables
        self.pnp_phase = 0
        self.pnp_sequence = [self.P1_approach, 
                             self.P1_pick, 
                             self.P1_approach,
                             self.home,
                             self.B1_approach,
                             self.B1_place,
                             self.B1_approach,
                             self.home,
                             self.P2_approach, 
                             self.P2_pick, 
                             self.P2_approach,
                             self.home,
                             self.B2_approach,
                             self.B2_place,
                             self.B2_approach,
                             self.home,
                             self.P3_approach, 
                             self.P3_pick, 
                             self.P3_approach,
                             self.home,
                             self.B3_approach,
                             self.B3_place,
                             self.B3_approach,
                             self.home,]
        self.pnp_gripper_states = [0, 1, 1, 1, 1, 0, 
                                   0, 0, 0, 1, 1, 1, 
                                   1, 0, 0, 0, 0, 1,
                                   1, 1, 1, 0, 0, 0]
        self.pnp_target = self.pnp_sequence[self.pnp_phase]
        self.jaka_interface.robot.power_on_gripper()
        self.jaka_interface.robot.open_gripper()
        time.sleep(2) # Wait for the gripper to change state

    def control_loop(self):
        q = np.array(self.jaka_interface.robot.joint_position)
        tcp = np.array(self.jaka_interface.robot.tcp_position)

        if np.allclose(tcp[:3], self.pnp_target[:3]):
            if self.pnp_phase < len(self.pnp_sequence) - 1:
                if self.pnp_gripper_states[self.pnp_phase]:
                    self.jaka_interface.robot.close_gripper()
                else:
                    self.jaka_interface.robot.open_gripper()
                time.sleep(0.5) # Wait for the gripper to change state
                self.pnp_phase += 1
                self.pnp_target = self.pnp_sequence[self.pnp_phase]

        q_des = self.compute_trajectory(tcp, self.pnp_target)
    
        qd_des = (q_des - q) / self.dt
        
        if self.obstacle_pos is not None:
            qd_safe = self.safe_controller(qd_des, tcp[:3], self.obstacle_pos)
            q_out = q + qd_safe * self.dt
        else:
            q_out = q_des
        
        self.jaka_interface.robot.servo_j(q_out, MoveMode.ABSOLUTE)
    

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