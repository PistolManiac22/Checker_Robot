import os

os.system("python checkers.py -f ab -s k --turn=2")

"""
-f ab : use AlphaBetaAgent as the first agent (o on the board, currently red caps)
-s k  : human player as the second agent (x on the board, currently blue caps)
--turn=2 : start the game with the second player

use ql for QLearningAgent, sl for SarsaLearningAgent, ssl for SarsaSoftmaxAgent

Pieces are currently placed on white grids, in both game engine and real life (left-bottom grid should be black)
x (blue) pieces are placed in the bottom side and o (red) pieces are placed in the top side
"""