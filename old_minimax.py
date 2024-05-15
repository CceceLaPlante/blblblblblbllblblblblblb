import numpy as np
class SB_states () :
  def __init__ (self,sizeX,sizeY) :
    self.sizeX = sizeX
    self.sizeY = sizeY

  def empty_state (self) :
    return (0,0,np.zeros((self.sizeY,2*self.sizeX-1),dtype = bool))

  def nb_to_coord (self,nb) :
    x = self.sizeX
    y = self.sizeY

    ligne = nb//x
    colonne= nb%x

    return colonne,ligne

  def get_co(self, state,colonne,ligne) :
    #colonne : coord x, ligne : coord y
    if colonne >= 2*self.sizeX -1 or ligne >= self.sizeY or colonne < 0 or ligne <0:
      # out of bound
      return False
    elif ligne == self.sizeY-1 and colonne >= self.sizeX -1 :
      # tackle the fact that in our state, we create vertical lines for the
      # last line, which shouldn't exist
      return False
    else :
      return state[ligne,colonne]


  def coord_to_nb (self,ligne,colonne) :
    return ligne*x+colonne


  def get_key (self, state) :
    y,x = state.shape
    if y > 7 :
      raise ValueError("it's too big to be packed into a uint8 number")
    return tuple(np.packbits(state,axis=0).flatten())

  def exploration (self,state,cache,explored,player_idx) :
    # player_idx: 0 or 1
    #iterator generating the possible next state, action to go to that state, and the explored states so far
    # cache : dictionary of known states: state_key : [minimax_value, nb_of_visit]
    # output : next state, explored nodes, and if the information have been found in the cach or not

    s = state[2]
    y,x = np.where(s==False)

    for i in range(len(x)) :
      if y[i].item() == self.sizeY-1 and x[i].item() >= self.sizeX -1 :
        continue
      points = state[player_idx]

      #check if playing [x,y] or it's variations would be in the cache :
      n_state = s.copy()
      n_state[y[i].item(),x[i].item()] = True
      lr_s = np.fliplr(n_state)
      ud_s = np.flipud(n_state)
      udlr_s = np.flipud(np.fliplr(n_state))

      variations = [
          self.get_key(n_state),
          self.get_key(ud_s),
          self.get_key(udlr_s),
          self.get_key(lr_s),
      ]

      for v in variations :
        if v in explored :
          continue
        if v in cache :
          cache[v][1] += 1
          yield cache[v],(x[i].item(),y[i].item()),explored, True

      explored.append(self.get_key(n_state))

      # find the closed boxes :
      up1,up2,d1,d2,l1,l2,r1,r2 = False,False,False,False,False,False,False,False


      # 1st case : move vertical, box to the right or to the left :
      if x[i].item() >= self.sizeX -1 :
        up1 = self.get_co(n_state,x[i].item()-(self.sizeX-1),y[i].item())
        up2 =  self.get_co(n_state,x[i].item()-(self.sizeX-2),y[i].item())
        # for down, it's the same but with y+1
        d1 = self.get_co(n_state,x[i].item()-(self.sizeX-1),y[i].item()+1)
        d2 = self.get_co(n_state,x[i].item()-(self.sizeX),y[i].item()+1)
        # left it's obviously true (it's the move we just made)
        l1 = True
        l2 = True

        r1 = self.get_co(n_state,x[i].item()+1,y[i].item())
        if x[i].item == self.sizeX-1 :
          r2 = False
        else :
          r2 = self.get_co(n_state,x[i].item()-1,y[i].item())

      # scd case : move horizontal : box to the up or bottom
      else :
        # bot : obviously true
        d1, d2 = True,True

        up1 = self.get_co(n_state, x[i].item(),y[i].item()+1 )
        up2 = self.get_co(n_state, x[i].item(),y[i].item()-1 )

        l1 = self.get_co(n_state, x[i].item()+(self.sizeX-1),y[i].item())
        l2 = self.get_co(n_state, x[i].item()+(self.sizeX-1),y[i].item()-1)

        r1 = self.get_co(n_state, x[i].item()+self.sizeX,y[i].item())
        r2 = self.get_co(n_state, x[i].item()+self.sizeX,y[i].item()-1)

      if up1 and d1 and l1 and r1 :
        points += 1
      if up2 and d2 and l2 and r2 :
        points += 1


      ns = [0,0,n_state]
      ns[player_idx] = points
      ns[1-player_idx] = state[1-player_idx]

      yield ((ns[0],ns[1],ns[2].copy()),(x[i].item(),y[i].item()), explored,False)

def minimax (state, maximizing_player_id,sb,explored,cache,current_player=0,depth = 0) :
  #check if state is terminal
  max_points = (sb.sizeX-1)*(sb.sizeY-1)//2

  if state[0] > max_points or state[1] > max_points or state[0]+state[1] == 2*max_points:
    cache[sb.get_key(state[2])] = state[maximizing_player_id] - state[1-maximizing_player_id]
    return state[maximizing_player_id] - state[1-maximizing_player_id],cache

  f= lambda x,y: x
  if current_player == maximizing_player_id :
    f = min
    m = float("+inf")
  else :
    f = max
    m = float("-inf")

  mm = state[maximizing_player_id] - state[1-maximizing_player_id]
  minimax_action = ()
  gen = sb.exploration(state,{}, [],current_player)
  i = 0
  for s,a,e,b in gen :
    if depth == 0 :
      print(i)
    if s[current_player] < state[current_player] :
      next_p = current_player
    else :
      next_p = 1-current_player
    mm,cache2 = minimax(s,maximizing_player_id, sb, [],{},next_p,depth+1)
    cache.update(cache2)

    if b :
      cache[sb.get_key(state[2])] = state[maximizing_player_id] - state[1-maximizing_player_id]
      print("found cache")
    if f(mm,m) == mm :
      minimax_action = a,s
      m = mm

    i+=1

  if depth == 0 :

    return m, minimax_action,cache

  if i == 0 :
    print("jvais mtuer")
    print(state)

  return m,cache

sb = SB_states(2,3)
cache = {}
explored = []
starting_state = sb.empty_state()

max_points = (sb.sizeX-1)*(sb.sizeY-1)//2
stop = lambda starting_state :  starting_state[0] > max_points or starting_state[1] > max_points or starting_state[0]+starting_state[1] == 2*max_points
i = 0
while not stop(starting_state) :
  print(starting_state)
  m,mma,cache = minimax(starting_state, 0,sb,explored,cache,current_player=i%2,depth = 0)
  i +=1
  starting_state = mma[1]
print(starting_state)


# test de gen

"""sb = SB_states(2,3)
st = sb.empty_state()[2]
st[2,0] = True
st[1,0] = True
st[1,1] = True
st = (0,0,st)
print(st)
print("=====")
for s,a,e,b in sb.exploration (st,{},[],0) :
  print(s)"""