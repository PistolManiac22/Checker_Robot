import logging
from .RobotArm import RobotArm
from .CoordinateTransformer import CoordinateTransformer


class Mover:
    def __init__(self, robot_is_first):
        self.transformer = CoordinateTransformer()
        self.robot_arm = RobotArm()
        self.robot_is_red = robot_is_first

    def take_picture(self):
        self.robot_arm.take_picture()

    def move_all(self, move, captured_board_pos, kinged_board_pos, center_positions, new_king_img_pos,
                 box_img_pos, board_angle, image_size):
        """
        Moves the piece with the robot arm according to the move, then captures pieces if necessary, kings a piece if
        necessary, and moves the robot arm back to the start position.
        :param move: move positions on the board ((start_x, start_y), (end_x, end_y))
        :param captured_board_pos: list of captured positions on the board [(x, y), ...]
        :param kinged_board_pos: kinged position (x, y) on the board or None
        :param center_positions: list of center positions (image coordinates) of the squares/pieces [[(x, y), ...], ...]
        :param new_king_img_pos: new king position (x, y) on the image
        :param box_img_pos: captured box center position (x, y) on the image
        :param board_angle: angle between the rows/cols and x-axis in degrees (0-90)
        :param image_size: (height, width) size of the image to calculate real coordinates from
        """
        logging.info(
            f"Moving piece from {move[0]} to {move[1]}, capturing {captured_board_pos}, kinging {kinged_board_pos}")
        if kinged_board_pos:
            self.do_king_move(move, kinged_board_pos, center_positions, new_king_img_pos, box_img_pos, board_angle,
                              image_size)
        else:
            self.do_move(move, center_positions, board_angle, image_size)
        self.do_captures(captured_board_pos, center_positions, box_img_pos, board_angle, image_size)
        self.move_to_start_position()

    def do_move(self, move, center_positions, board_angle, image_size):
        origin_board, target_board = move
        origin_real = self.transformer.board_xy_to_real_xy(origin_board, center_positions, image_size)
        target_real = self.transformer.board_xy_to_real_xy(target_board, center_positions, image_size)
        self.robot_arm.move_piece_from_to(origin_real, target_real, board_angle)

    def do_captures(self, captured_board_positions, center_positions, box_img, board_angle, image_size):
        for capt_board in captured_board_positions:
            capt_real = self.transformer.board_xy_to_real_xy(capt_board, center_positions, image_size)
            box_real = self.transformer.img_xy_to_real_xy(box_img, image_size)
            self.robot_arm.move_piece_from_to_and_drop(capt_real, box_real, board_angle)

    def do_king_move(self, move, kinged_board, center_positions, king_img, box_img, board_angle, image_size):
        origin_board, _ = move
        origin_real = self.transformer.board_xy_to_real_xy(origin_board, center_positions, image_size)
        kinged_real = self.transformer.board_xy_to_real_xy(kinged_board, center_positions, image_size)
        king_real = self.transformer.img_xy_to_real_xy(king_img, image_size)
        box_real = self.transformer.img_xy_to_real_xy(box_img, image_size)
        self.robot_arm.move_piece_from_through_to_and_drop(origin_real, [kinged_real], box_real, board_angle)
        self.robot_arm.move_piece_from_to(king_real, kinged_real, board_angle)

    def move_to_start_position(self):
        self.robot_arm.move_to_start_position()