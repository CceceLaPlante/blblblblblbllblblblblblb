import pyspiel 
import numpy as np

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

def get_ret(state):
    # a very ugly function...
    os = state.observation_string(0)
    ret1 = 0
    ret2 = 0
    for s in os : 
        if s == '0':
            ret1 += 1
        elif s == '1':
            ret2 += 1

    return [ret1,ret2]

def _minimax(state, maximizing_player_id, row, col, alpha, beta, cache):
    if state.is_terminal():
        ret = get_ret(state)
        return ret[maximizing_player_id]-ret[1-maximizing_player_id]
    
    player = state.current_player()
    if player == maximizing_player_id :
        best_value = float('-inf')

        for action in state.legal_actions() :
            val = _minimax(state.child(action), maximizing_player_id, row, col, alpha, beta, cache)
            best_value = max(best_value, val)
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break

        return best_value
    
    else:
        best_value = float('inf')

        for action in state.legal_actions() :
            val = _minimax(state.child(action), maximizing_player_id, row, col, alpha, beta, cache)
            best_value = min(best_value, val)
            beta = min(beta, best_value)
            if beta <= alpha:
                break

        return best_value

def minimax_search ( game, 
                    state= None,
                    maximizing_player_id = 0,
                    state_to_key=lambda state : state) :

    if state is None:
        state = game.new_initial_state()

    row = game.get_parameters()["num_rows"]
    col = game.get_parameters()["num_cols"]
    cache = {}
    alpha = float('-inf')
    beta = float('inf')
    best_action = None
    
    action_val = [_minimax(state.child(action),maximizing_player_id,row,col,alpha,beta,cache) for action in state.legal_actions()]
    
    if maximizing_player_id == state.current_player():
        best_action = state.legal_actions()[np.argmax(action_val)]

    else:
        best_action = state.legal_actions()[np.argmin(action_val)]

    return best_action
    
def main():
    game = pyspiel.load_game("dots_and_boxes(num_rows=2,num_cols=2)")
    state = game.new_initial_state()
    print(minimax_search(game, state, 0))
    while not state.is_terminal():
        print(state)
        state.apply_action(minimax_search(game, state, 0))
    print(state)

    state = game.new_initial_state()
    while not state.is_terminal():
        print(state)
        if state.current_player() == 0:
            state.apply_action(minimax_search(game, state, 0))
        else:
            a = input("Enter your action: ")
            state.apply_action(int(a))



if __name__ == "__main__":
    main()

