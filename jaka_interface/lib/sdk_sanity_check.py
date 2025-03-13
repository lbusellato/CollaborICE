#!/usr/bin/env python3.10
# TODO: would be really useful to integrate all of this into a series of unit tests
import jkrc
import time

INCR = 1

Enable = True
Disable = False

robot = jkrc.RC("10.5.5.100")
ret = robot.login()
ret = robot.power_on() #power on
ret = robot.enable_robot()
print(robot.get_robot_status())

#while True:
#    ret = robot.get_joint_position()
#    joint_state = ret[1]
#
#    tcp = robot.get_tcp_position()
#    tcp = tcp[1]
#ret = robot.get_joint_position()
#joint_state = ret[1]
#
#tcp = robot.get_tcp_position()
#tcp = tcp[1]
#
#start = time.time()
#for i in range(100):
#    ret = robot.kine_forward(joint_state)
#print(f"elapsed: {(time.time() - start)/100}")
#
#start = time.time()
#for i in range(100):
#    ret = robot.kine_inverse(joint_state, tcp)
#print(f"elapsed: {(time.time() - start)/100}")
#
robot.servo_move_enable(True)

for i in range(200):
    robot.servo_j([0.001,0 , 0, 0, 0, 0], INCR, step_num=1)

for i in range(200):
    robot.servo_j([-0.001,0 , 0, 0, 0, 0], INCR, step_num=1)

#tcp_pose = robot.get_tcp_position()
#tcp_pose = tcp_pose[1]
#
#import copy
#import numpy as np
#
#target_pose = copy.deepcopy(tcp_pose)
#target_pose[2] = target_pose[2] - 50
#
#tcp_pose = np.array(tcp_pose)
#target_pose = np.array(target_pose)
#
#n_waypoints = 1
#waypoints = [tcp_pose * (1-t) + target_pose * t for t in np.linspace(0, 1, n_waypoints)]
#
#for i in range(n_waypoints):
#    joint_state = robot.get_joint_position()
#    next_pose = robot.kine_inverse(joint_state[1], waypoints[i])
#    
#    robot.servo_j(next_pose[1], 0, 1)
#
#tcp_pose = robot.get_tcp_position()
#tcp_pose = tcp_pose[1]
#
#target_pose = copy.deepcopy(tcp_pose)
#target_pose[2] = target_pose[2] + 50
#
#tcp_pose = np.array(tcp_pose)
#target_pose = np.array(target_pose)
#
#waypoints = [tcp_pose * (1-t) + target_pose * t for t in np.linspace(0, 1, n_waypoints)]
#
#for i in range(n_waypoints):
#    joint_state = robot.get_joint_position()
#    next_pose = robot.kine_inverse(joint_state[1], waypoints[i])
#    
#    robot.servo_j(next_pose[1], 0, 1)

