def create_matrix_sym(row, col):
  list_matrix = []

  nb_line = row*(col + 1) + col*(row + 1)

  #axial Symetry
  transposition_matrix = np.zeros((nb_line, nb_line))
  for i in range(col*(row + 1)):
    x = (i//col)
    y = i%col
    transposition_matrix[i, x*col + (col - 1 - y)] = 1
  for j in range(row*(col + 1)):
    x = (j//(col + 1))
    y = j%(col + 1)
    transposition_matrix[col*(row + 1) + j, col*(row + 1) + x*(col + 1) + (col + 1 - j - 1)] = 1

  list_matrix.append(transposition_matrix.copy())

  transposition_matrix = np.zeros((nb_line, nb_line))
  for i in range(col*(row + 1)):
    x = (i//col)
    y = i%col
    transposition_matrix[i, (row + 1 - 1 - x)*col + y] = 1
  for j in range(row*(col + 1)):
    x = (j//(col + 1))
    y = j%(col + 1)
    transposition_matrix[col*(row + 1) + j, col*(row + 1) + (row - 1 - x)*(col + 1) + y] = 1

  list_matrix.append(transposition_matrix.copy())

  #Corner Symetry

  transposition_matrix = np.eye(nb_line, nb_line)
  corner1 = 0
  corner2 = col*(row + 1)
  transposition_matrix[corner1,corner1] = 0
  transposition_matrix[corner2,corner2] = 0
  transposition_matrix[corner1,corner2] = 1
  transposition_matrix[corner2,corner1] = 1

  list_matrix.append(transposition_matrix.copy())

  transposition_matrix = np.eye(nb_line, nb_line)
  corner1 = col -1
  corner2 = col*(row + 1) + col
  transposition_matrix[corner1,corner1] = 0
  transposition_matrix[corner2,corner2] = 0
  transposition_matrix[corner1,corner2] = 1
  transposition_matrix[corner2,corner1] = 1

  list_matrix.append(transposition_matrix.copy())

  transposition_matrix = np.eye(nb_line, nb_line)
  corner1 = col*row
  corner2 = col*(row + 1) + (col + 1)*(row - 1)
  transposition_matrix[corner1,corner1] = 0
  transposition_matrix[corner2,corner2] = 0
  transposition_matrix[corner1,corner2] = 1
  transposition_matrix[corner2,corner1] = 1

  list_matrix.append(transposition_matrix.copy())

  transposition_matrix = np.eye(nb_line, nb_line)
  corner1 = col*(row + 1)  - 1
  corner2 = col*(row + 1) + (col + 1) *row - 1
  transposition_matrix[corner1,corner1] = 0
  transposition_matrix[corner2,corner2] = 0
  transposition_matrix[corner1,corner2] = 1
  transposition_matrix[corner2,corner1] = 1

  list_matrix.append(transposition_matrix.copy())


  return list_matrix




def update_dictionnary( dic, val, move_number, board, list_matrix):
    board_set = set()
    board_set.add(tuple(board))
    for matrix in list_matrix:
        current_board = board_set.copy()
        for board in current_board:
            new_board = np.matmul(np.array(board), matrix)
            board_set.add(tuple(new_board))

    if (len(dic[move_number]) == 0):
      print("first move of size " + str(move_number) + " reach")
    for board in board_set:
      dic[move_number][board] = val



def _minimax_up(state, maximizing_player_id, row, col, dic, alpha, beta, list_matrix):
    """
    Implements a min-max algorithm

    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.

    Returns:
      The optimal value of the sub-game starting in state
    """

    points1 = state.to_string().count("1")
    points2 = state.to_string().count("2")

    if state.is_terminal():
        return points1



    #Get the dictionnary of the state with the same number of move
    move_number = state.move_number()

    board = np.zeros(row*(col + 1) + col*(row + 1))

    if move_number != 0 :
        move_already_made = state.information_state_string().split(",")
        move_already_made =  list(map(int, move_already_made))

        board[move_already_made] = 1

    player = state.current_player()


    same_move = dic.get(move_number)

    if (same_move == None):
        dic[move_number] = {}
    else:
        val = same_move.get(tuple(board))


        if val != None:

            if player == maximizing_player_id:
                # print(state,points1 + val)
                return points1 + val

            else:
                # print(state,row*col - points2 - val)
                return row*col - points2 - val


    if player == maximizing_player_id:
        maxVal = 0

        for action in state.legal_actions():
           val = _minimax_up(state.child(action), maximizing_player_id, row, col, dic,  alpha, beta, list_matrix)
           maxVal = max(val, maxVal)
           alpha = max(val, alpha)
           if beta <= alpha:
              break

        if beta > alpha:
            update_dictionnary( dic, maxVal - points1, move_number, board, list_matrix)

        return maxVal

    else:
        minVal = row*col + 1

        for action in state.legal_actions():
           val = _minimax_up(state.child(action), maximizing_player_id, row, col, dic,  alpha, beta, list_matrix)
           minVal = min(val, minVal)
           beta = min(val, beta)
           if beta <= alpha:
              break


        if beta > alpha:
            update_dictionnary( dic, row*col - minVal - points2, move_number, board, list_matrix)

        return minVal





def minimax_search_up(game,
                   state=None,
                   maximizing_player_id=None,
                   state_to_key=lambda state: state):
    """Solves deterministic, 2-players, perfect-information 0-sum game.

    For small games only! Please use keyword arguments for optional arguments.

    Arguments:
      game: The game to analyze, as returned by `load_game`.
      state: The state to run from.  If none is specified, then the initial state is assumed.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN. The default (None) will suppose the player at the root to be
        the MAX player.

    Returns:
      The value of the game for the maximizing player when both player play optimally.
    """
    game_info = game.get_type()

    if game.num_players() != 2:
        raise ValueError("Game must be a 2-player game")
    if game_info.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("The game must be a Deterministic one, not {}".format(
            game.chance_mode))
    if game_info.information != pyspiel.GameType.Information.PERFECT_INFORMATION:
        raise ValueError(
            "The game must be a perfect information one, not {}".format(
                game.information))
    if game_info.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("The game must be turn-based, not {}".format(
            game.dynamics))
    if game_info.utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("The game must be 0-sum, not {}".format(game.utility))

    row = game.get_parameters()["num_rows"]
    col = game.get_parameters()["num_cols"]
    #Instancing Dictionnary
    dic = {}

    list_matrix = create_matrix_sym(row, col)

    if state is None:
        state = game.new_initial_state()
    if maximizing_player_id is None:
        maximizing_player_id = state.current_player()
    v = _minimax_up(
        state.clone(),
        maximizing_player_id,
        row, col, dic, -1, col*row + 1, list_matrix)
    if v > row*col/2:
      winner = 1
    elif v == row*col/2:
      winner = 0
    else :
      winner = -1
    return winner


def upgrad(num_row,num_col):
    games_list = pyspiel.registered_names()
    assert "dots_and_boxes" in games_list
    game_string = "dots_and_boxes(num_rows=" + str(num_row) + ",num_cols="+ str(num_col) + ")"

    print("Creating game: {}".format(game_string))
    game = pyspiel.load_game(game_string)

    value = minimax_search_up(game)

    if value == 0:
        print("It's a draw")
    else:
        winning_player = 1 if value == 1 else 2
        print(f"Player {winning_player} wins.")