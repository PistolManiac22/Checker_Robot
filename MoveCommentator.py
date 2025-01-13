import logging
import sys
import numpy as np
import subprocess

"""
The prompt we give gemini will change depending on how you want the functions to look like

This one adds everything togethers, meaning the human player will receive recommended move, own move and motivation all at the same time:
You help a human player playing checkers against you. You do not ask the player for their recommendations.
If you receive a recommended move, present it to the player, your opponent, clearly ensuring your tone is warm and encouraging.
In addition, sometimes you need to congratulate the player on a move they made, congratulate them with positive reinforcement, acknowledging their skill and progress.
Sometimes to give the player more interaction, you also tell the player the move you are making.
Keep the interaction supportive, fun, and engaging, so the player feels motivated and enjoys the game.\n",

This one will work with the three seperate functions I wrote
"You help a human player playing checkers against you. You do not ask the player for their recommendations.
Based on your input, you will either do one of these three:
1. provide helpful, friendly, and constructive advice based on moves recommended by a SARSA agent. When a recommended move is ready, present it to the player, your opponent, clearly ensuring your tone is warm and encouraging.
2. You will comment on your opponents move. If your opponent captures one of your pieces, congratulate them with positive reinforcement, acknowledging their skill and progress.
3. Tell the player your own move.

Most importantly, keep the interaction supportive, fun, and engaging, so the player feels motivated and enjoys the game.\n",
"""


class MoveCommentator:
    def __init__(self, robot_is_first, motivation_level, recommended_move_level, own_move_level):
        self.process = None
        self.robot_is_first = robot_is_first
        self.motivation_rounds = self._init_rounds(motivation_level)
        self.recommended_move_rounds = self._init_rounds(recommended_move_level)
        self.own_move_rounds = self._init_rounds(own_move_level)
        self.rounds_from_last_motivation = np.random.randint(3)
        self.rounds_from_last_recommended_move = np.random.randint(3)
        self.rounds_from_last_own_move = np.random.randint(3)
        self.last_board_state = None

    def _init_rounds(self, level):
        """
        level -> value
        1 -> sys.maxsize (max int value)
        2 -> 4
        3 -> 3
        4 -> 2
        5 -> 1
        """
        if level == 1:
            return sys.maxsize
        return 6 - level

    def comment(self, move, opponents_move, opponents_best_moves, board_state):
        """
        Comments on the robots next move and opponents last move.
        :param move: Next move done by robot.
        :param opponents_move: Last move done by opponent (included in board_state) - ((start_x, start_y), (end_x, end_y))
        :param opponents_best_moves: 1-3 top moves that the opponent could have done from the last state. [((start_x, start_y), (end_x, end_y)), ...]
        :param board_state: Current state of the board.
        """
        self.rounds_from_last_motivation += 1
        self.rounds_from_last_recommended_move += 1
        self.rounds_from_last_own_move += 1

        prompt = ""
        if self.rounds_from_last_motivation >= self.motivation_rounds:
            self.rounds_from_last_motivation = 0
            # put maybe the whole board state analysis here since that is what oponents move basically does
            # prompt += (f" Opponent did this move: {opponents_move}. First board state: {board_state1}. Second consecutive board state: {board_state2}.")
            prompt += (f"Opponent did this move: {opponents_move}. You do not tell the coordinates of the move. ")
            print("Opponents (players) move: ", opponents_move)
        if self.rounds_from_last_recommended_move >= self.recommended_move_rounds:
            self.rounds_from_last_recommended_move = 0
            prompt += (f"Recommened move for the player: {opponents_best_moves}. ")
            print("Recommendated moves: ", opponents_best_moves)
        if self.rounds_from_last_own_move >= self.own_move_rounds:
            self.rounds_from_last_own_move = 0
            prompt += (f"The move you are making: {move}. You do not tell the coordinates of the move.")
            print("Own move made: ", move)
        if prompt.strip() == "":
            logging.info("no prompt this turn")
            return
        prompt += (f"As context: Previous board state: {self.last_board_state}. Current board state: {board_state}.")
        logging.info(f"Prompt: {prompt}")
        self.last_board_state = board_state
        self.generate_and_read_comment(prompt)

    def generate_and_read_comment(self, text):
        self.process = subprocess.Popen(
            [f"/home/robotics/workspace2/team9_ws/run_commentator.sh \"{text}\""],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True
        )

    def wait_till_reading_finishes(self):
        if self.process is None:
            return
        try:
            stdout, stderr = self.process.communicate(timeout=20)
            if stderr != "":
                logging.error(stderr)
            logging.info(stdout)
        except:
            logging.error("Commentator failed")
        finally:
            self.process = None


def test():
    commentator = MoveCommentator(1, 1, 1, 1)
    opponents_best_moves = "((0, 3), (1, 4)"
    move = "((5, 6), (4, 5)"
    opponents_move = "((2, 6), (3, 5))"
    captures = 3

    board_state1 = [[1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 1, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 2, 0, 2, 0, 2],
                    [2, 0, 2, 0, 2, 0, 2, 0],
                    [0, 2, 0, 2, 0, 2, 0, 2]]

    board_state2 = [[1, 0, 1, 0, 1, 0, 1, 0],
                    [0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 2, 0, 2, 0, 2],
                    [2, 0, 2, 0, 2, 0, 2, 0],
                    [0, 2, 0, 2, 0, 2, 0, 2]]

    commentator.comment(move, opponents_move, opponents_best_moves, board_state1, board_state2)

if __name__ == "__main__":
    test()