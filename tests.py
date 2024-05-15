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

def in_cache(state,cache,sizeX,sizeY) :
    variations = get_variations(state,sizeX,sizeY)

    for v in variations :
        if v in cache :
            return True,cache[v]
    return False,None

game = pyspiel.load_game("dots_and_boxes")
state = game.new_initial_state()
state.apply_action(0)
state.apply_action(1)

print(state)
print(get_key(state,2,2))
cache = {}
cache[get_key(state,2,2)] = 1


similary_state = game.new_initial_state()
similary_state.apply_action(4)
similary_state.apply_action(5)
print(similary_state)
print(get_key(similary_state,2,2))
print(in_cache(similary_state,cache,2,2))

print(state)