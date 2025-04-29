import rclpy

from rclpy.node import Node
from jaka_interface.interface import JakaInterface
from jaka_interface.data_types import MoveMode

class SimTestNode(Node):
    def __init__(self):
        self.interface = JakaInterface(simulated=True)

        self.interface.initialize()
        self.interface.robot.enable_servo_mode()
        print(self.interface.robot.get_joint_position())

        self.interface.robot.servo_j([0.1,0,0,0,0,0], MoveMode.INCREMENTAL)

        print(self.interface.robot.get_joint_position())

        

        super().__init__('sim_test_node')
    
def main():
    rclpy.init()
    node = SimTestNode()    
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()

if __name__=='__main__':
    main()