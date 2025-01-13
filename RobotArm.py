import rclpy
from tm_msgs.msg import *
from tm_msgs.srv import *
import logging
import time


class RobotArm:
    def __init__(self):
        self.base = 115
        self.safe = self.base + 35
        self.safe_box = self.base + 90
        self.dropping_margin = 5
        self.start_position = (350, 350, 730)
        self.start_orientation = (-180, 0, 90)
        self.slow_down = 15

    def move_to_start_position(self):
        logging.info(f"Moving to base position")
        self.move_to_all_param(self.start_position[0], self.start_position[1], self.start_position[2],
                               self.start_orientation[0], self.start_orientation[1], self.start_orientation[2])

    def move_piece_from_to(self, origin, target, angle):
        """
        Move a piece from origin to target. Goes directly above the origin, ends above the target. Expects to move just
        above the board - won't get above boxes
        :param origin: (origin_x, origin_y)
        :param target: (target_x, target_y)
        :param angle: angle to rotate the gripper by from the start rotation clockwise
        """
        self.move_piece_from_through_to(origin, [], target, angle)

    def move_piece_from_through_to(self, origin, through, target, angle):
        """
        Move a piece from origin to target, touch all the through squares along the way. Goes directly above the origin,
        ends above the target. Expects to move just above the board - won't get above boxes
        :param target: (origin_x, origin_y)
        :param through: [(through1_x, through1_y), ...]
        :param origin: (target_x, target_y)
        :param angle: angle to rotate the gripper by from the start rotation clockwise
        """
        self.grab_from_and_touch(origin, through, angle)
        self.move_to_pos_and_angle(target[0], target[1], self.safe, angle)
        self.move_to_pos_and_angle(target[0], target[1], self.base + self.dropping_margin, angle, self.slow_down)
        self.open_gripper()
        self.move_to_pos_and_angle(target[0], target[1], self.safe, angle, self.slow_down)

    def move_piece_from_through_to_and_drop(self, origin, through, target, angle):
        """
        Move a piece from origin to target, touch all the through squares, and drop it at target from a safe height.
        Goes directly above the origin, ends above the target.
        :param origin: (origin_x, origin_y)
        :param through: [(through1_x, through1_y), ...]
        :param target: (target_x, target_y)
        :param angle: angle to rotate the gripper by from the start rotation clockwise
        """
        self.grab_from_and_touch(origin, through, angle)
        if len(through) > 0:
            self.move_to_pos_and_angle(through[-1][0], through[-1][1], self.safe_box, angle, self.slow_down)
        else:
            self.move_to_pos_and_angle(origin[0], origin[1], self.safe_box, angle, self.slow_down)
        self.move_to_pos_and_angle(target[0], target[1], self.safe_box, angle)
        self.open_gripper()

    def grab_from_and_touch(self, origin, through, angle, safe_speed_override = 100):
        self.move_to_pos_and_angle(origin[0], origin[1], self.safe, angle, safe_speed_override)
        self.move_to_pos_and_angle(origin[0], origin[1], self.base, angle, self.slow_down)
        self.close_gripper()
        self.move_to_pos_and_angle(origin[0], origin[1], self.base, angle - 20)
        self.move_to_pos_and_angle(origin[0], origin[1], self.base, angle)
        self.move_to_pos_and_angle(origin[0], origin[1], self.safe, angle, self.slow_down)
        for th in through:
            self.move_to_pos_and_angle(th[0], th[1], self.safe, angle)
            self.move_to_pos_and_angle(th[0], th[1], self.base + self.dropping_margin, angle, self.slow_down)
            self.move_to_pos_and_angle(th[0], th[1], self.safe, angle, self.slow_down)

    def move_piece_from_to_and_drop(self, origin, target, angle):
        """
        Move a piece from origin to target and drop it from a safe height.
        Goes directly above the origin, ends above the target.
        :param target: (origin_x, origin_y)
        :param origin: (target_x, target_y)
        :param angle: angle to rotate the gripper by from the start rotation clockwise
        """
        self.move_to_pos_and_angle(origin[0], origin[1], self.safe_box, angle)
        self.grab_from_and_touch(origin, [], angle, self.slow_down)
        self.move_to_pos_and_angle(origin[0], origin[1], self.safe_box, angle, self.slow_down)
        self.move_to_pos_and_angle(target[0], target[1], self.safe_box, angle)
        self.open_gripper()

    def move_to(self, x, y, z):
        self.move_to_pos_and_angle(x, y, z, 0)

    def move_to_pos_and_angle(self, x, y, z, angle=0, speed_percent=100):
        logging.info(f"Moving to ({x}, {y}, {z}) with angle {angle}")
        self.move_to_all_param(x, y, z, self.start_orientation[0], self.start_orientation[1],
                               self.start_orientation[2] - angle, speed_percent)

    def move_to_all_param(self, x, y, z, a, b, c, speed_percent=100):
        position = f"{x}, {y}, {z}, {a}, {b}, {c}"
        script = f"PTP(\"CPP\",{position},{speed_percent},100,0,false)"
        self.send_script(script)

    def open_gripper(self):
        logging.info("Opening gripper")
        self.set_io(0.0)

    def close_gripper(self):
        logging.info("Closing gripper")
        self.set_io(1.0)

    def take_picture(self):
        logging.info("Taking picture")
        self.send_script("Vision_DoJob(job1)")

    def send_script(self, script):
        arm_node = rclpy.create_node('arm')
        arm_cli = arm_node.create_client(SendScript, 'send_script')
        while not arm_cli.wait_for_service(timeout_sec=1.0):
            arm_node.get_logger().info('service not availabe, waiting again...')
        move_cmd = SendScript.Request()
        move_cmd.script = script
        arm_cli.call_async(move_cmd)
        arm_node.destroy_node()

    def set_io(self, state):
        gripper_node = rclpy.create_node('gripper')
        gripper_cli = gripper_node.create_client(SetIO, 'set_io')
        while not gripper_cli.wait_for_service(timeout_sec=1.0):
            node.get_logger().info('service not availabe, waiting again...')
        io_cmd = SetIO.Request()
        io_cmd.module = 1
        io_cmd.type = 1
        io_cmd.pin = 0
        io_cmd.state = state
        gripper_cli.call_async(io_cmd)
        gripper_node.destroy_node()
