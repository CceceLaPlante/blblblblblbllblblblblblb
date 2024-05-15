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

ROTATION_MATRIX = create_matrix_sym(2,2)


def get_optimal_state(state,init_state) :
    h = np.array(state.history(),dtype=int)
    opti_state = init_state.copy()
    for i in h :
        opti_state[i] = 1
    
    return opti_state
    

def get_ret(state):
    # a very ugly function...
    os = state.observation_string(0)
    ret1 = 0
    ret2 = 0
    for s in os : 
        if s == '1':
            ret1 += 1
        elif s == '2':
            ret2 += 1

    return [ret1,ret2]

def get_key(state,init_state):
    ot = get_optimal_state(state,init_state)
    ot = np.array(ot,dtype=bool)
    key = tuple(np.packbits(ot))
    return key

def is_similar(state1, state2,init_state):
    ot1 = get_optimal_state(state1,init_state)

    if get_key(state1,init_state) == get_key(state2,init_state) :
        return True

    for matrix in ROTATION_MATRIX : 
        ot1_rotated = np.matmul(ot1, matrix)
        key = tuple(np.packbits(np.array(ot1_rotated,dtype=bool)))
        if key == get_key(state2,init_state) :
            return True
    return False

def in_cache(state1, cache,init_state):
    ot1 = get_optimal_state(state1,init_state)
    k = get_key(state1,init_state)
    if k in cache.keys() :
        return cache[k]

    for matrix in ROTATION_MATRIX : 
        for key_cache in cache.keys():
            ot1_rotated = np.matmul(ot1, matrix)
            key = tuple(np.packbits(np.array(ot1_rotated,dtype=bool)))
            if key == key_cache :
                return cache[key]
        

    return None



def _minimax(state, maximizing_player_id, row, col, alpha, beta, cache,init_state):
    if state.is_terminal() :
        return state.returns()[maximizing_player_id], cache
        
    
    player = state.current_player()
    if player == maximizing_player_id :
        best_value = float('-inf')

        for action in state.legal_actions() :
            sc = state.child(action)
            cached = sc.is_terminal()
            inc = in_cache(sc, cache,init_state)
            if inc is not None:
                val = inc
                cached = True
                c = {}
            else:
                val,c = _minimax(sc, maximizing_player_id, row, col, alpha, beta, cache,init_state)
            best_value = max(best_value, val)
            alpha = max(alpha, best_value)
            if beta < alpha:
                break
            elif alpha == beta :
                return best_value,cache
                
            else :
                if cached :
                    cache.update(c)
                    cache[get_key(state,init_state)] = val

        return best_value,cache
    
    else:
        best_value = float('inf')

        for action in state.legal_actions() :

            sc = state.child(action)
            cached = sc.is_terminal()
            inc = in_cache(sc, cache,init_state)
            if inc is not None:
                val = inc
                cached = True
                c = {}
                
            else:
                val,c = _minimax(sc, maximizing_player_id, row, col, alpha, beta, cache,init_state)
                cache.update(c)
            best_value = min(best_value, val)
            beta = min(beta, best_value)
            if beta < alpha:
                break
            elif alpha == beta :
                return best_value,cache
            else :
                cache.update(c)
                cache[get_key(state,init_state)] = val

        return best_value,cache

def minimax_search ( game, 
                    state= None,
                    maximizing_player_id = 0) :

    if state is None:
        state = game.new_initial_state()

    row = game.get_parameters()["num_rows"]
    col = game.get_parameters()["num_cols"]
    cache = {}
    alpha = float('-inf')
    beta = float('inf')
    init_state = np.zeros((row*(col + 1) + col*(row + 1)))
    
    return _minimax(state, maximizing_player_id, row, col, alpha, beta, cache,init_state)[0]

    
def main():
    game = pyspiel.load_game("dots_and_boxes(num_rows=2,num_cols=2)")
    print(minimax_search(game, None))


if __name__ == "__main__":
    main()

