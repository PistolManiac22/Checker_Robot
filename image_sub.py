#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from tm_msgs.msg import *
from tm_msgs.srv import *
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import time
import logging
import threading
import sys
import tkinter as tk
from tkinter import ttk
import random
from rclpy.task import Future

sys.path.append("/home/robot/colcon_ws/install/tm_msgs/lib/python3.6/site-packages")

from .Mover import Mover
from .MoveCommentator import MoveCommentator
from .ImageDetector import ImageDetector
from .GameEngine import GameEngine
from .RobotArm import RobotArm

# Global Variables for the UI (Do not burn me tomas, there is no other way)
global robot_is_first_global, mot_level_global, rec_level_global, own_move_level_global
robot_is_first_global = True
mot_level_global = 3
rec_level_global = 3
own_move_level_global = 3

class ImageSub(Node):
    def __init__(self, nodeName, robot_is_first, motivation_level, recommended_move_level, own_move_level):
        super().__init__(nodeName)
        self.future = Future()

        savedir = "./calibrate/values/"
        self.cam_mtx = np.load(savedir + "cam_mtx.npy")
        self.newcam_mtx = np.load(savedir + "newcam_mtx.npy")
        self.dist = np.load(savedir + "dist.npy")
        self.roi = np.load(savedir + "roi.npy")

        self.mover = Mover(robot_is_first)
        # robot is first -> robot is red -> player is blue -> player is 2
        self.game_engine = GameEngine(2 if robot_is_first else 1)
        self.commentator = MoveCommentator(robot_is_first, motivation_level, recommended_move_level, own_move_level)

        self.robot_is_first = robot_is_first  # = robot is red

        self.img_number = 0
        self.time_between_action = 1
        self.subscription = self.create_subscription(
            Image, "techman_image", self.image_callback, 10
        )

    def image_callback(self, data):
        logging.info("Received image")
        bridge = CvBridge()
        raw_image = bridge.imgmsg_to_cv2(data)
        image = self.undistort_and_crop(raw_image)
        # self.show_image_wait_for_key(image)
        # self.show_and_save_image(image)
        # self.take_another_image()
        # time.sleep(1)
        self.act_based_on_image(image)

    def undistort_and_crop(self, raw_image):
        image = cv.undistort(raw_image, self.cam_mtx, self.dist, None, self.newcam_mtx)
        x, y, w, h = self.roi
        return image[y: y + h, x: x + w]

    def show_image_wait_for_key(self, image):
        cv.imshow("image", image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def show_and_save_image(self, image):
        cv.imshow("image", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
        cv.imwrite("./img_" + str(self.img_number) + ".jpg", image)
        self.img_number += 1

    def act_based_on_image(self, image):
        detector = ImageDetector(image, self.robot_is_first, 0.1)
        if not detector.can_extract_board_state():
            logging.error(detector.get_error())
            self.take_another_image()
            return
        board = detector.get_board_state()
        logging.info("detected board state")
        self.game_engine.print_board(board)
        if not self.game_engine.is_board_state_valid(board):
            logging.error(self.game_engine.get_error())
            self.take_another_image()
            return
        if not self.game_engine.has_board_state_changed(board):
            logging.info("Board state has not changed.")
            self.take_another_image()
            return
        if self.game_engine.did_player_win(board):
            print("Game ended, you won. Congratulations!")
            self.future.set_result(True)
            return
        opponents_move = self.game_engine.get_last_opponents_move(board)
        opponents_best_moves = self.game_engine.get_last_opponents_best_moves()
        move, is_winning_move, captured_board_pos, kinged_board_pos = (
            self.game_engine.make_move(board)
        )
        self.commentator.comment(move, opponents_move, opponents_best_moves, board)
        center_positions = detector.get_centers()
        king_position = detector.get_new_king_position()
        captured_box_position = detector.get_captured_box_pos()
        board_angle = detector.get_board_angle()
        self.mover.move_all(
            move,
            captured_board_pos,
            kinged_board_pos,
            center_positions,
            king_position,
            captured_box_position,
            board_angle,
            image.shape[:2]
        )
        self.commentator.wait_till_reading_finishes()
        if is_winning_move:
            print("Game ended, you lost. Better luck next time!")
            self.future.set_result(True)
            return
        self.take_another_image()
        # DON'T DO ANYTHING ELSE HERE, RETURN FROM THE CALLBACK

    def take_another_image(self):
        """
        ALWAYS RETURN FROM THE PREVIOUS CALLBACK AFTER CALLING THIS
        Takes another image after self.time_between_action time.
        """
        time.sleep(self.time_between_action)
        self.mover.take_picture()


def prompt_user_if_he_wants_start():
    while True:
        user_input = input("Do you want to start? (y/n/r - random): ")
        if user_input == "y" or user_input == "Y":
            return True
        elif user_input == "n" or user_input == "N":
            return False
        elif user_input == "r" or user_input == "R":
            return bool(np.random.randint(2))
        else:
            print('Invalid input. Please enter "y", "n" or "r".')


def prompt_user_for_interaction_level(type_of_interaction):
    while True:
        user_input = input(
            f"Enter the level of robot's comments about {type_of_interaction} 1-5 (1 = no interaction, 5 = interaction every turn):")
        if user_input.isdigit() and 1 <= int(user_input) <= 5:
            return int(user_input)
        else:
            print("Invalid input. Please enter a number between 1 and 5.")


def start_robots_action():  # aka send_script.py
    logging.info("Starting robots action - send_script.py")
    time.sleep(1)
    robot_arm = RobotArm()
    robot_arm.move_to_start_position()
    robot_arm.take_picture()
    logging.info("Robots action ended - send_script.py")


def run_robot(robot_is_first, motivation_level, recommended_move_level, own_move_level):
    logging.info("Started listening to images")
    rclpy.init(args=None)
    t1 = threading.Thread(target=start_robots_action)
    t1.start()
    node = ImageSub("image_sub", robot_is_first, motivation_level, recommended_move_level, own_move_level)
    rclpy.spin_until_future_complete(node, node.future)
    node.destroy_node()
    t1.join()
    rclpy.shutdown()

def main():
    logging.basicConfig(level=logging.INFO)
    global robot_is_first_global, mot_level_global, rec_level_global, own_move_level_global
    def CenterWindowToDisplay(Screen: tk.Tk, width: int, height: int, scale_factor: float = 1.0):
        """Centers the window to the main display/monitor"""
        screen_width = Screen.winfo_screenwidth()
        screen_height = Screen.winfo_screenheight()
        x = int(((screen_width / 2) - (width / 2)) * scale_factor)
        y = int(((screen_height / 2) - (height / 1.5)) * scale_factor)
        return f"{width}x{height}+{x}+{y}"

    def update_choice(choice, variable_name):
        global robot_is_first_global, mot_level_global, rec_level_global, own_move_level_global
        if variable_name == "robot_is_first_global":
            if choice == "Blue (Robot First)":
                robot_is_first_global = True
            elif choice == "Red (You are First)":
                robot_is_first_global = False
            elif choice == "Random":
                robot_is_first_global = random.choice([True, False])
        elif variable_name == "mot_level_global":
            mot_level_global = int(choice)
        elif variable_name == "rec_level_global":
            rec_level_global = int(choice)
        elif variable_name == "own_move_level_global":
            own_move_level_global = int(choice)

    app = tk.Tk()
    app.geometry(CenterWindowToDisplay(app, 900, 600, 1.0))
    app.title("Let's Play Checkers!")
    app.configure(bg="#00008b")

    # Add a frame for better grouping and design
    main_frame = tk.Frame(app, bg="#00008b", padx=20, pady=20, relief="groove", bd=5)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    title = tk.Label(main_frame, text="Let's Play Checkers!", font=('Arial', 36, 'bold'), fg="#1e90ff", bg="#00008b")
    title.pack(pady=20)

    text1 = tk.Label(main_frame, text="What color do you want to be?", font=('Arial', 16), bg="#00008b", fg="white")
    text1.pack(pady=10)
    optionmenu_var1 = tk.StringVar(value="Blue (Robot First)")
    optionmenu1 = ttk.OptionMenu(main_frame, optionmenu_var1, "Blue (Robot First)", "Blue (Robot First)", "Red (You are First)", "Random",
                                  command=lambda choice: update_choice(choice, "robot_is_first_global"))
    optionmenu1.pack(pady=5)

    text2 = tk.Label(main_frame, text="Level of motivating comments", font=('Arial', 16), bg="#00008b", fg="white")
    text2.pack(pady=10)
    optionmenu_var2 = tk.StringVar(value="3")
    optionmenu2 = ttk.OptionMenu(main_frame, optionmenu_var2, "3", "1", "2", "3", "4", "5",
                                  command=lambda choice: update_choice(choice, "mot_level_global"))
    optionmenu2.pack(pady=5)

    text3 = tk.Label(main_frame, text="Level of comments about recommended moves for you", font=('Arial', 16), bg="#00008b", fg="white")
    text3.pack(pady=10)
    optionmenu_var3 = tk.StringVar(value="3")
    optionmenu3 = ttk.OptionMenu(main_frame, optionmenu_var3, "3", "1", "2", "3", "4", "5",
                                  command=lambda choice: update_choice(choice, "rec_level_global"))
    optionmenu3.pack(pady=5)

    text4 = tk.Label(main_frame, text="Level of comments about robot's moves", font=('Arial', 16), bg="#00008b", fg="white")
    text4.pack(pady=10)
    optionmenu_var4 = tk.StringVar(value="3")
    optionmenu4 = ttk.OptionMenu(main_frame, optionmenu_var4, "3", "1", "2", "3", "4", "5",
                                  command=lambda choice: update_choice(choice, "own_move_level_global"))
    optionmenu4.pack(pady=5)

    startbutton = tk.Button(main_frame, text="Start Game", command=app.destroy, bg="#ff4500", fg="white", font=('Arial', 16, 'bold'), relief="raised")
    startbutton.pack(pady=20)

    # Add a footer
    footer = tk.Label(app, text="Enjoy your game!", font=('Arial', 12), bg="#00008b", fg="white")
    footer.pack(side="bottom", pady=10)

    app.mainloop()
    # following runs after button press (app.destroy)
    run_robot(robot_is_first_global, mot_level_global, rec_level_global, own_move_level_global)

if __name__ == "__main__":
    main()
