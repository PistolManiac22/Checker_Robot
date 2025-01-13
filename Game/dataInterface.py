import os
# from Piece_Detection import PieceDetect
import copy

legalMoves = []
currentMove = []
lastBoard = []
currentBoard = []
capturedPieces = []
kingedPiece	= []
img_path = "./TestImage/img_12.jpg" # The path of the raw image you want to use as input

def interfacePlayerMove():
	"""
	Get player's move from image and check if it's illegal
	If legal, then send the move into the game engine
	Else reset the board
	"""
	legal = False

	while legal == False :
		"""
		GetCurrentBoard()
		ret = interfaceCalculateMove()
		if ret == None :
			n = input("Press enter to process player's move")
			continue
		start, end = ret[0], ret[1]
		"""
		n = input("enter the move : ")
		start, end = [int(n.split(' ')[0]), int(n.split(' ')[1])], [int(n.split(' ')[2]), int(n.split(' ')[3])]
		move = [start[0], start[1], end[0], end[1]]
		# n = input("Press enter to process player's move")
		# print(legalMoves)
		for legalmove in legalMoves :
			if legalmove[0] == start and legalmove[1] == end :
				legal = True
				break

		if legal == False : # When player did an illegal move
			# return interfaceResetBoard()
			print("Illegal move")
			continue

	"""
	legal = False

	while legal == False :
		GetCurrentBoard()
		ret = interfaceCalculateMove()
		if ret == None :
			n = input("Press enter to process player's move")
			continue

		start, end = ret[0], ret[1]
		move = [start[0], start[1], end[0], end[1]]
		n = input("Press enter to process player's move")
		for legalmove in legalMoves :
			if legalmove[0] == start and legalmove[1] == end :
				legal = True
				break

		if legal == False : # When player did an illegal move
			interfaceResetBoard()
	"""

	print("Player's move : " + str(start) + " to " + str(end))

	return move

def setLastBoard(board):
	global lastBoard
	lastBoard = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
	for i in range(8):
		for j in range(4):
			lastBoard[i][j] = board[i][j*2] + board[i][j*2+1]

def CheckLegalMove(board):
	legal = False
	while legal == False :
		ret = interfaceCalculateMove(board)
		if ret == None :
			return 0
		start, end = ret[0], ret[1]
		for legalmove in legalMoves :
			if legalmove[0] == start and legalmove[1] == end :
				legal = True
				break
		if legal == 0 : # When player did an illegal move
			return 1
	return 2

def interfaceAIMove(move):
	"""
	Get AI's move
	Once AI made its move, this function will be called
	"""
	global currentMove
	currentMove = move
	start = move[0]
	end = move[1]
	#print("Computer's move : " + str(start) + " to " + str(end))

	# TODO : Send the move to robot arm and make the move

def interfaceCapture(pieces, act):
	"""
	If act is AI, pieces need to be taken
	"""
	global capturedPieces
	print("Piece in position " + str(pieces) + " is captured by " + act)
	if act == "AI" :
		capturedPieces.append(pieces)
		print(capturedPieces)

	# TODO : Send the position to the arm and take those pieces away

def interfaceKinged(pieces, act):
	"""
	If the act is AI, pieces need to be kinged
	"""
	global kingedPiece
	if act == "AI" :
		print(currentMove)
		if lastBoard[currentMove[0][0]][currentMove[0][1]] == 1 and pieces == currentMove[1]:
			print("Piece in position " + str(pieces) + " is kinged by " + act)
			kingedPiece.append(pieces)
			# return pieces

	# return None

def returnKingedPiece() :
	if kingedPiece == [] :
		return None
	return kingedPiece[0]

def clearKingedPiece():
	global kingedPiece
	kingedPiece = []

def returncapturedPieces():
	return capturedPieces

def clearcapturedPieces():
	print("Cleared")
	global capturedPieces
	capturedPieces = []

def returnLastBoard():
	return lastBoard

def returnCurrentMove():
	return currentMove

def returnCurrentBoard():
	return currentBoard

def interfaceGetLegalMove(moves):
	"""
	Get player's legal moves from engine
	"""
	global legalMoves
	legalMoves = moves

def interfaceGetLastBoard(board):
	"""
	Record the board state as the last state for later use
	"""
	global lastBoard
	lastBoard = board

def interfaceResetBoard():
	"""
	Do something to reset the board to the last move (lastBoard) using robot arm
	What if it cannot be reset ? e.g. player hide the piece
	"""
	print("Illegal move ! Please reset the board !")
	return False

def MatrixTransform(mat) :
	"""
	Transform the input 8*8 matrix into a corresponding 8*4 matrix that can be processed by the engine
	"""
	global currentBoard
	currentBoard = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
	for i in range(8):
		for j in range(4):
			currentBoard[i][j] = mat[i][j*2] + mat[i][j*2+1]

def GetCurrentBoard():
	"""
	Main function to call to take a picture and get current board state
	"""
	# TODO : Take a picture here
	# MatrixTransform(PieceDetect(img_path))

def CheckTakenPieces(mat):
	"""
	Check if the pieces player took away are all correct
	"""
	while True:
		legal = True
		# GetCurrentBoard()
		for i in range(8):
			for j in range(4):
				if mat[i][j] != currentBoard[i][j] :
					legal = False
					print("Illegal move ! Please take the pieces away correctly !")
					n = input("Press enter to continue")
		if legal == True :
			return

def interfaceCalculateMove(board):
	"""
	Find out player's move and if it's illegal
	"""
	findStart, findEnd, anyMove = False, False, False
	for i in range(8):
		for j in range(4):
			if lastBoard[i][j] == 2 and board[i][j] == 0 :
				if findStart == True :
					return interfaceResetBoard()
				start = [i, j]
				findStart = True
			if lastBoard[i][j] == 0 and board[i][j] == 2 :
				if findEnd == True :
					return interfaceResetBoard()
				end = [i, j]
				findEnd = True
			if lastBoard[i][j] == 0 and board[i][j] == 1 :
				return interfaceResetBoard()
			if lastBoard[i][j] != board[i][j] :
				anyMove = True
	if findStart == True and findEnd == True :	# Only move one player's own piece, then continue to check
		return start, end
	elif anyMove == False : # Hasn't move anything
		return None
	else :	# Basically involves all the other illegal moves
		return None
	return None

# python checkers.py -f ab -s k