import numpy as np
import pyspiel

def empty_state (sizeX,sizeY) :
    return np.zeros((sizeY,2*sizeX-1),dtype = bool)

def get_co(state,colonne,ligne) :
    sizeY,sizeX = state.shape
    #colonne : coord x, ligne : coord y
    if colonne >= 2*sizeX -1 or ligne >= sizeY or colonne < 0 or ligne <0:
      # out of bound
      return False
    elif ligne == sizeY-1 and colonne >= sizeX -1 :
      # tackle the fact that in our state, we create vertical lines for the
      # last line, which shouldn't exist
      return False
    else :
      return state[ligne,colonne]
    
def apply_action (state, action) :
    sizeY,sizeX = state.shape
    ligne= action // sizeX
    colonne= action % sizeX
    
    ns = state.copy()
    ns[ligne,colonne] = True
    return ns

def get_key (state,val1,val2) :
    y,x = state.shape
    if y > 7 :
      raise ValueError("it's too big to be packed into a uint8 number")
    return ((val1,val2),tuple(np.packbits(state,axis=0).flatten()))

def in_cache(state,cache,val1,val2) :
    lr_s = np.fliplr(state)
    ud_s = np.flipud(state)
    udlr_s = np.flipud(np.fliplr(state))

    variations = [
        get_key(state,val1,val2),
        get_key(ud_s,val1,val2),
        get_key(udlr_s,val1,val2),
        get_key(lr_s,val1,val2),
    ]
    for v in variations :
        if v in cache :
            return True,cache[v]
    return False,None

def _minimax (state,opti_state, maximizing_player_id,cache):
    """
    Implements a min-max algorithm

    Arguments:
      state: The current state node of the game.
      maximizing_player_id: The id of the MAX player. The other player is assumed
        to be MIN.

    Returns:
      The optimal value of the sub-game starting in state
    """

    if state.is_terminal():
        cache[get_key(state)] = state.player_return(maximizing_player_id)
        print(state,state.returns())    
        return state.player_return(maximizing_player_id),cache,False
    
    player = state.current_player()
    if player == maximizing_player_id:
        selection = max
    else:
        selection = min

    actions = state.legal_actions()
    states = [state.child(action) for action in actions]
    #states_opti = [get_state_opti(s) for s in states]
    i = 0
    useless = []
    for so in states :
        # in cache ?
        b,v = in_cache(so,cache,val1,val2)
        if b :
            return v,cache, True

        # redundancy check
        variations = [
            get_key(so),
            get_key(np.fliplr(so)),
            get_key(np.flipud(so)),
            get_key(np.flipud(np.fliplr(so))),  
        ]
        j = 0
        for s in states[:i] :

            if get_key(s) in variations :
                useless.append(i)
            j+=1
        i+=1

    values_children = [_minimax(states[i],maximizing_player_id,cache) for i in range(len(states)) if i not in useless]
    true_value_children = []
    i=0
    for vc, c, cached in values_children :
        val1,val2 = states[i].player_return(maximizing_player_id),states[i].player_return(1-maximizing_player_id)
        cache.update(c)
        if cached :
            cache[get_key(states_opti[i],val1,val2)] = vc
        true_value_children.append(vc)
        i = i+1


    return selection(true_value_children),cache,False

def play_minimax (state,opti_state,maximizing_player_id,cache) :
    states = [state.child(action) for action in state.legal_actions()]
    states_opti = [apply_action(opti_state,action) for action in state.legal_actions()]
    actions = state.legal_actions()

    i = 0
    m = float("-inf")
    for i in range(len(states)) :
        v,c,_ = _minimax(states[i],states_opti[i], maximizing_player_id,cache)
        if v > m :
            m = v
            action = actions[i]
            cache.update(c)
    return action,cache


# tests :
game_string = "dots_and_boxes(num_rows=3,num_cols=3)"
game = pyspiel.load_game(game_string)
state = game.new_initial_state()
# [!] it's the nb of dots, not of rows and col, so it's 3x3 and not 2x2
opti_state = empty_state(4,4)

print("============")
print("test of the _minimax function")
# test of the _minimax function
v,c,_ = _minimax(state,opti_state,0,{})
print("============")
print(v)

print("============")

# let's play against the AI
print("============")
cache = {}
"""while not state.is_terminal() :
    print(state)
    if state.current_player() == 0 :   
        action,cache = play_minimax(state.clone(),opti_state,0,cache)
        print("i play : ",action)
        state.apply_action(action)
        opti_state = apply_action(opti_state,action)
        
        print(state)
    if state.current_player() == 1 :
        print("your move : ")
        a = np.random.choice(state.legal_actions())
        print(a)
        state.apply_action(int(a))
        opti_state = apply_action(opti_state,int(a))
        print("============")

    if state.is_terminal() :
        break

"""
winrate = [0,0]
for i in range (100) :
    state = game.new_initial_state()
    while not state.is_terminal() :
        if state.current_player() == 0 :   
            action,cache = play_minimax(state.clone(),opti_state,0,cache)
            state.apply_action(action)
            opti_state = apply_action(opti_state,action)
            
        if state.current_player() == 1 :
            a = np.random.choice(state.legal_actions())
            state.apply_action(int(a))
            opti_state = apply_action(opti_state,int(a))

        if state.is_terminal() :
            break
    if state.player_return(0) > state.player_return(1) :
        winrate[0] += 1
    else :
        winrate[1] += 1

print(winrate)
    