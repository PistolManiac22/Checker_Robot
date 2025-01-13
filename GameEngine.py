import logging
from .Game.checkers import *
from .Game.agents import *
class GameEngine():
    """
    Board state is always represented by an 8x8 2D array with the following values meanings:
        0: empty square
        1: red piece
        2: blue piece
        3: red king
        4: blue king
    """
    def __init__(self, playerpiece = 1):
        self.last_state = self._get_starting_board() # initial board state
        self.sec_last_state = self._get_rotated_board(self.last_state)
        self.rotated = False
        self.first_move = (playerpiece == 2)
        self.init_set = True
        self.error = None
        self.AIAgent = AlphaBetaAgent(depth = 3)
        self.PlayerPiece = playerpiece
        self.AIPiece = (playerpiece*2)%3
        self.currentState = GameState(prev_state=None, the_player_turn=(playerpiece == 1))
        self.last_player_move = None

    def _get_starting_board(self):
         return [[1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 2, 0, 2, 0, 2],
                [2, 0, 2, 0, 2, 0, 2, 0],
                [0, 2, 0, 2, 0, 2, 0, 2]]
         #return [[1, 0, 1, 0, 1, 0, 1, 0],
         #       [0, 1, 0, 0, 0, 0, 0, 1],
         #       [0, 0, 1, 0, 1, 0, 1, 0],
         #       [0, 1, 0, 1, 0, 0, 0, 0],
         #       [0, 0, 0, 0, 2, 0, 0, 0],
         #       [0, 2, 0, 0, 0, 0, 0, 2],
         #       [2, 0, 2, 0, 2, 0, 2, 0],
         #       [0, 2, 0, 2, 0, 2, 0, 0]]

    def _get_rotated_board(self, board):
         return [row[::-1] for row in board[::-1]]

    def get_error(self):
        return self.error

    def update_game_state(self, board, turn):
        """
        This is currently for debugging, board is input with (y coordinate, x coordinate) system
        """
        self.currentState.board.spots = Eight2Four(board)
        self.currentState.board.player_turn = turn

    def set_last_state(self, board):
        """
        This is currently for debugging, board is input with (y coordinate, x coordinate) system
        """
        self.last_state = board

    def print_board(self, board):
        for i in range(8):
            logging.info(board[i])
        logging.info("")

    def is_board_state_valid(self, board):
        """
        Checks if the board state (= last move) is valid.
        :param board: 8x8 array of pieces on the board
        :return: True if the board state is valid, False otherwise
        """
        if self.first_move :
            self.init_set = False
            if board == self.sec_last_state:
                 self.rotated = True
                 self.update_game_state(rotate_board(board, self.rotated), not (self.PlayerPiece == 1))
                 return True
            if board != self.last_state:
                self.error = "Initial board state is not valid. Make sure that top left black corner has a piece."
                return False
            self.update_game_state(rotate_board(board, self.rotated), not (self.PlayerPiece == 1))
            return True
        if self.init_set :
            if board == self.sec_last_state:
                self.rotated = True
                self.init_set = False
                self.update_game_state(rotate_board(board, self.rotated), not (self.PlayerPiece == 2))
            if board == self.last_state:
                self.init_set = False
                self.update_game_state(rotate_board(board, self.rotated), not (self.PlayerPiece == 2))
        board = rotate_board(board, self.rotated)
        board = GetProperBoard(self.last_state, board, self.PlayerPiece)
        self.last_player_move = CalculateMove(self.last_state, board, self.PlayerPiece)
        if self.last_player_move == None :
            self.error = "The board state is not valid. Please make a valid move."
            return False
        move = [Eight2FourCoord(self.last_player_move[0]), Eight2FourCoord(self.last_player_move[1])]
        legal = False
        for legalMove in self.currentState.get_legal_actions() :
            if self._get_direct_move(legalMove) == move :
                move = legalMove
                legal = True
                break
        if not legal :
            self.error = "You did an illegal move. Please make a valid move."
            return False
        proper_state = self.currentState.generate_successor(action = move, switch_player_turn=False) # not sure about this line
        if self.undo_kings(Four2Eight(proper_state.board.spots)) == self.undo_kings(board): # If the board state is valid etc., we know that we will continue so let's save the board state
            self.last_state = Four2Eight(proper_state.board.spots)
            self.currentState = proper_state
            self.currentState.board.player_turn = not self.currentState.board.player_turn
            return True
        self.error = "You did not take away all the captured pieces or you took wrong pieces. Please take them correctly."
        return False

    def _get_direct_move(self, move):
         return [move[0], move[-1]]


    def undo_kings(self, board):
        res = []
        for i in range(8):
            tmp = []
            for j in range(8):
                if board[i][j] > 2:
                    tmp.append(board[i][j] - 2)
                else:
                    tmp.append(board[i][j])
            res.append(tmp)
        return res

    def has_board_state_changed(self, board):
        """
        Checks if the board state has changed since the last move.
        :param board: 8x8 array of pieces on the board
        :return: True if the board state has changed, False otherwise
        """
        # because of saving the state in is_board_state_valid, this function's code does not make sense
        """
        if self.first_move :
             return True
        board = rotate_board(board, self.rotated)
        #self.last_state = Four2Eight(self.last_state)
        if self.last_state == GetProperBoard(self.last_state, board, self.PlayerPiece) :
            return False
        """
        return True

    def did_player_win(self, board):
        """
        Checks if the game has ended.
        :param board: 8x8 array of pieces on the board
        :return: True if the game has ended, False otherwise
        """
        return self.currentState.get_legal_actions() == []

    def get_last_opponents_move(self, board):
        """
        Extracts the last move of the opponent from the board state.
        :param board: 8x8 array of pieces on the board
        :return: ((start_x, start_y), (end_x, end_y))
        """
        board = rotate_board(board, self.rotated)
        if self.last_player_move == None :
            return None
        move = [rotate_move(self.last_player_move[0], self.rotated), rotate_move(self.last_player_move[1], self.rotated)]
        return ((move[0][1], move[0][0]), (move[1][1], move[1][0]))

    def get_last_opponents_best_moves(self):
        """
        Return 1-3 top moves that the opponent could have done from the last state.
        :return: [((start_x, start_y), (end_x, end_y)), ...]
        """
        if self.first_move:
             return []
        state2cal = GameState(prev_state = None, the_player_turn=(self.PlayerPiece==1))
        state2cal.board.spots = Eight2Four(self.last_state)
        calABAgent = AlphaBetaAgent(depth = 3)
        calQAgent = QLearningAgent()
        calSAgent = SarsaSoftmaxAgent()
        moveAB = calABAgent.get_action(state2cal)
        moveQ = calQAgent.get_action(state2cal)
        moveS = calSAgent.get_action(state2cal)
        moves = [moveAB, moveQ, moveS]
        new_moves = []
        for move in moves:
             if move not in new_moves:
                  new_moves.append(move)
        ret = []
        for i in range(len(new_moves)):
            m, n = Four2EightCoord([new_moves[i][0][0],new_moves[i][0][1]]),Four2EightCoord([new_moves[i][1][0],new_moves[i][1][1]])
            if self.rotated:
                ret.append(((7-m[1], 7-m[0]),(7-n[1], 7-n[0])))
            else:
                ret.append(((m[1], m[0]),(n[1], n[0])))
        return ret

    def make_move(self, board):
        """
        Makes a move based on the board state and outputs the move, boolean whether it is the last winning move,
        captured positions and kinged position.
        :param board: 8x8 array of pieces on the board
        :return: (((start_x, start_y), (end_x, end_y)), is_winning_move, [(capt_1_x, capt_1_y), (capt_2_x, capt_2_y), ...], (kinged_x, kinged_y)), no capture: [], no king: None
        """
        if self.first_move:
             self.first_move = False
        board = rotate_board(board, self.rotated)
        self.currentState.board.player_turn = not (self.PlayerPiece == 1)
        move = self.AIAgent.get_action(self.currentState)
        self.last_state = GetProperBoard(self.last_state, board, self.PlayerPiece)

        self.currentState = self.currentState.generate_successor(action = move, switch_player_turn = False)
        new_board = Four2Eight(self.currentState.board.spots)
        caps = GetCapturedPieces(self.last_state, new_board, self.PlayerPiece, self.rotated)
        king = GetKingedPiece(self.last_state, new_board, self.AIPiece, [Four2EightCoord(move[0]), Four2EightCoord(move[-1])], self.rotated)
        move = [Four2EightCoord(move[0]), Four2EightCoord(move[-1])]
        isWinning = True
        for i in range(8):
            for j in range(8):
                if new_board[i][j] == self.PlayerPiece :
                    isWinning = False
        board = Four2Eight(new_board)
        self.currentState.board.player_turn = not self.currentState.board.player_turn
        self.last_state = new_board

        if self.currentState.get_legal_actions() == []:
            isWinning = True
        if self.rotated:
            return (((7-move[0][1], 7-move[0][0]), (7-move[-1][1], 7-move[-1][0])), isWinning, caps, king)
        return (((move[0][1], move[0][0]), (move[-1][1], move[-1][0])), isWinning, caps, king)

def rotate_board(board, rot):
     if not rot :
          return board
     ret = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
     for i in range(8):
        for j in range(8):
            ret[i][j] = board[7-i][7-j]
     return ret

def rotate_move(move, rot):
     if not rot :
          return move
     return [7-move[0], 7-move[1]]

def CoordSwitch(board):
    for i in range(8):
        for j in range(i, 8):
            tmp = board[i][j]
            board[i][j] = board[j][i]
            board[j][i] = tmp
    return board

def Eight2Four(board) :
    ret = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i in range(8):
        for j in range(4):
            ret[i][j] = board[i][j*2] + board[i][j*2+1]
    return ret

def Four2Eight(board):
    ret = [[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
    for i in range(8):
        for j in range(8):
            if (i+j)%2 == 0 :
                ret[i][j] = board[i][int(j/2)]
            else :
                ret[i][j] = 0
    return ret

def Four2EightCoord(coord):
    odd = (coord[0]%2 == 1)
    return [coord[0], coord[1]*2 + odd]

def Eight2FourCoord(coord):
     return [coord[0], int(coord[1]/2)]

def CalculateMove(lastBoard, board, Piece):
	"""
	Find out player's move and if it's illegal (8*8)
	"""
	findStart, findEnd = False, False
	for i in range(8):
		for j in range(8):
			if (lastBoard[i][j] == Piece and board[i][j] == 0 ) or (lastBoard[i][j] == Piece + 2 and board[i][j] == 0 ):
				if findStart == True :
					return None
				start = [i, j]
				findStart = True
			if (lastBoard[i][j] == 0 and board[i][j] == Piece) or (lastBoard[i][j] == 0 and board[i][j] == Piece + 2) :
				if findEnd == True :
					return None
				end = [i, j]
				findEnd = True
			if lastBoard[i][j] == 0 and board[i][j] == (Piece*2)%3 :
				return None
	if findStart == True and findEnd == True :	# Only move one player's own piece, then continue to check
		return start, end
	else :	# Basically involves all the other illegal moves
		return None

def GetCapturedPieces(lastBoard, board, playerPiece, rot) :
    """
    Find the pieces that are captured by AI's move (8*8)
    """
    ret = []
    for i in range(8):
        for j in range(8):
            if lastBoard[i][j] == playerPiece or lastBoard[i][j] == playerPiece + 2 :
                if board[i][j] == 0 :
                    if rot:
                        ret.append((7-j, 7-i))
                    else:
                        ret.append((j,i))
    return ret

def GetKingedPiece(lastBoard, board, AIPiece, move, rot) :
    """
    Find the piece that is promoted by AI (8*8)
    """
    if board[move[1][0]][move[1][1]] == AIPiece + 2 and lastBoard[move[0][0]][move[0][1]] == AIPiece:
        if rot:
            return (7-move[1][1], 7-move[1][0])
        else:
            return (move[1][1], move[1][0])
    return None

def GetProperBoard(lastBoard, board, Piece):
    """
    Get the proper board state (with proper kinged pieces) from last state (with proper kinged pieces) and current board (without kinged pieces)
    8*8
    """
    moveKing = False
    temp = [0,0]
    for i in range(8):
        for j in range(8):
            if lastBoard[i][j] >= 3 and board[i][j] > 0 and board[i][j] <= 2:
                board[i][j] += 2
            if lastBoard[i][j] == (Piece + 2) and board[i][j] == 0 :
                moveKing = True
            if lastBoard[i][j] == 0 and board[i][j] == Piece :
                temp = [i, j]
    if moveKing :
        board[temp[0]][temp[1]] += 2
    return board