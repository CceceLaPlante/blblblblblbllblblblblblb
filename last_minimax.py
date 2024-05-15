import numpy as np
import pyspiel

def empty_state (sizeX,sizeY) :
    # sizeX : number of rows, sizeY : number of columns
    return np.zeros((2*(sizeX+1)-1,sizeY+1),dtype = bool)

def apply_action (state, action,sizeX,sizeY) :

    if action >= (sizeX+1)*sizeY :
        # vertical line
        x = (action - (sizeX+1)*sizeY) // sizeY
        y = (action - (sizeX+1)*sizeY) % sizeY
    else :
        # horizontal line
        x = action // sizeY
        y = action % sizeY
    ns = state.copy()
    ns[x,y] = True
    return ns

def get_key (state,sizeX,sizeY) :
    # generate a state :
    s = empty_state(sizeX,sizeY)
    l_action = [int(i) for i in state.information_state_string(0).split(", ")]
    for a in l_action :
        s = apply_action(s,a,sizeX,sizeY)
    val1 = state.returns()[0]
    
    
    return (val1,tuple(np.packbits(s,axis=0).flatten()))

def get_key_(state,val) :
    return (val,tuple(np.packbits(state,axis=0).flatten()))

def get_variations (state,sizeX,sizeY) :
    s = empty_state(sizeX,sizeY)

    l_action = [int(i) for i in state.information_state_string(0).split(", ")]
    for a in l_action :
        s = apply_action(s,a,sizeX,sizeY)
    val = state.returns()[0]
    variations = [
        get_key_(s,val),
        get_key_(np.flipud(s),val),
        get_key_(np.fliplr(s),val),
        get_key_(np.flipud(np.fliplr(s)),val),
    ]
    return variations
def get_variations (state,sizeX,sizeY) :
    return []
def in_cache(state,cache,sizeX,sizeY) :
    variations = get_variations(state,sizeX,sizeY)

    for v in variations :
        if v in cache :
            return True,cache[v]
    return False,None

def _minimax (state, maximizing_player_id, cache,sizeX,sizeY) :

    if state.is_terminal() :
        cache[get_key(state,sizeX,sizeY)] = state.returns()[maximizing_player_id]
        #print("terminal state")
        #print(state, state.returns()[maximizing_player_id])
        return state.returns()[maximizing_player_id], cache, False

    player = state.current_player()
    if player == maximizing_player_id:
        selection = max
    else:
        selection = min

    actions = state.legal_actions()
    states = [state.child(action) for action in actions]

    i = 0
    useless = []
    for s in states : 
        """b,v = in_cache(s,cache,sizeX,sizeY)
        if b :
            return v,cache, True"""
        
        variations = get_variations(s,sizeX,sizeY)
        j = 0
        for s2 in states[:i]+states[i+1:] :
            if get_key(s2,sizeX,sizeY) in variations :
                useless.append(j)
            j+=1
        i+=1
    values_children = [_minimax(states[i],maximizing_player_id,cache,sizeX,sizeY) for i in range(len(states)) if i not in useless]
    true_value_children = [states[i].rewards()[maximizing_player_id] for i in range(len(states)) if i not in useless]

    i=0

    for vc, c, cached in values_children :
        cache.update(c)
        if cached :
            cache[get_key(states[i],sizeX,sizeY)] = vc
        true_value_children[i] += vc
        i = i+1
    return selection(true_value_children),cache,False

def play_minimax (state,maximizing_player_id,cache,sizeX,sizeY) :
    actions = state.legal_actions()
    states = [state.child(action) for action in actions]

    i = 0
    m = float("-inf")
    for s in states :
        v ,c,_= _minimax(s,maximizing_player_id,cache,sizeX,sizeY)
        cache.update(c)
        if v > m :
            m = v
            action = actions[i]
        i+=1
    return action, cache

sizeX = 2
sizeY = 2
game_string = "dots_and_boxes(num_rows="+str(sizeX)+",num_cols="+str(sizeY)+")"
game = pyspiel.load_game(game_string)
state = game.new_initial_state()
# [!] it's the nb of dots, not of rows and col, so it's 3x3 and not 2x2
opti_state = empty_state(sizeX,sizeY)
cache = {}

while not state.is_terminal() :
    print(state)
    if state.current_player() == 0 :   
        action,cache = play_minimax(state.clone(),0,cache,sizeX,sizeY)
        print("i play : ",action)
        state.apply_action(action)
        
        print(state)
    if state.current_player() == 1 :
        print("your move : ")
        a = int(input(">>> "))
        print(a)
        state.apply_action(int(a))
        print("============")

    if state.is_terminal() :
        break
    
