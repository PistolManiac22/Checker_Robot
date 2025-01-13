#!/usr/bin/env python
import rclpy
from .RobotArm import RobotArm
import sys
sys.path.append('/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages')

def main(args=None):
    rclpy.init(args=args)
    robot_arm = RobotArm()
    robot_arm.move_to_start_position()
    robot_arm.take_picture()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
