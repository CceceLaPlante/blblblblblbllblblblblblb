import pyspiel
import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags
import sys
import os

from tqdm import tqdm

from open_spiel.python.egt.utils import game_payoffs_array


FLAGS = flags.FLAGS
flags.DEFINE_string("game_name","prisoners_dilemma","Game to play (prisoners_dilemma, subsidy_game, coordination_game, anti_coordination_game, battle_of_the_sex, biased_rps)")
flags.DEFINE_integer("max_iter",1000,"Number of iterations to run.")
flags.DEFINE_float("abs_tol",1e-4,"Absolute tolerance for convergence.")
flags.DEFINE_integer("h",10,"grid size for the starting population vector")
flags.DEFINE_boolean("arrows",True,"Display arrows")
flags.DEFINE_float("alpha",0.01,"learning rate")
flags.DEFINE_integer("factor",10,"factors for make the arrow plot less dense")

def get_game (game_name) :
    m_coordination_game = np.array([[1,0],[0,1]])
    m_anti_coordination_game = 1 - m_coordination_game


    m_s1 = np.array([[12,11],[0,10]])
    m_s2 = m_s1.T

    m_brps = np.array([
        [0,-0.05,0.25],
        [0.05,0,-0.5],
        [-0.25,0.5,0]
    ])

    m_rps = np.array( [
        [0,-1,1],
        [1,0,-1],
        [-1,1,0]
    ] )

    # this is the "positive" version, the negative version make openspiel diverging (why ?)
    m_pd1 = np.array([
        [3,0],
        [4,1]
    ])

    m_pd2 = m_pd1.T

    m_sb1 = np.array([
        [3,0],
        [0,2]
    ])

    m_sb2  = np.array([
        [2,0],
        [0,3]
    ])

    if game_name == "prisoners_dilemma" :
        return pyspiel.create_matrix_game(m_pd1,m_pd2),2
    elif game_name == "subsidy_game" :
        return pyspiel.create_matrix_game(m_s1,m_s2),2
    elif game_name == "coordination_game" :
        return pyspiel.create_matrix_game(m_coordination_game,m_coordination_game),2
    elif game_name == "anti_coordination_game" :
        return pyspiel.create_matrix_game(m_anti_coordination_game,m_anti_coordination_game),2
    elif game_name == "battle_of_the_sex" :
        return pyspiel.create_matrix_game(m_sb1,m_sb2),2
    elif game_name == "biased_rps" :
        return pyspiel.create_matrix_game (m_brps,m_brps),3

#from open_spiel, for more detail see open_spiel/python/algorithms/dyn.py
def replicator(state, fitness):
  avg_fitness = state.dot(fitness)
  return state * (fitness - avg_fitness)

class SinglePopulationDynamics(object):


  def __init__(self, payoff_matrix, dynamics):
    assert payoff_matrix.ndim == 3
    assert payoff_matrix.shape[0] == 2
    #assert np.allclose(payoff_matrix[0], payoff_matrix[1].T)
    self.payoff_matrix = payoff_matrix[0]
    self.dynamics = dynamics

  def __call__(self, state=None, time=None):

    state = np.array(state)
    assert state.ndim == 1
    assert state.shape[0] == self.payoff_matrix.shape[0]
    # (Ax')' = xA'
    fitness = np.matmul(state, self.payoff_matrix.T)
    return self.dynamics(state, fitness)
  
def replicator_dynamics (game,h,max_iter,max_tol, alpha, arrows, arrived, n,factor=10,game_name = "",axis = (0,1)) :


  axx = axis[0]
  axy = axis[1]

  payoff_matrix = game_payoffs_array(game)
  dyn = SinglePopulationDynamics(payoff_matrix,replicator)
  print(payoff_matrix)

  def zero_in_y(y,max_tol) :
    for i in y.flatten() :
      if i < max_tol :
        return True
    else :
      return False

  stop = lambda i,dy,y: i >= max_iter or dy <= max_tol or zero_in_y(y,max_tol)

  if n == 2 :
    l = np.linspace(1/h,1-(1/h),h)
    xs,ys=np.meshgrid(l,l)
    xs = xs.reshape((h,h,-1)) # add a new axis
    ys = ys.reshape((h,h,-1))

    Xs = np.concatenate((xs,ys), axis=2)
    Xs = Xs.reshape((h**n,n))

  elif  n == 3 :
    """l = np.linspace(1/h,1-(1/h),h)
    xs,ys,zs=np.meshgrid(l,l,l)
    xs = xs.reshape((h,h,-1)) # add a new axis
    ys = ys.reshape((h,h,-1))
    zs = zs.reshape((h,h,-1))

    Xs = np.concatenate((xs,ys,zs), axis=2)
    Xs = Xs.reshape((h**n,n))"""
    Xs = ((np.mgrid[0:h,0:h,0:h]/(h+1)).T).reshape(h**n,n)

  z_arrived = []
  x_arrived = []
  y_arrived = []

  Ys = []
  k = 0
  print("computing dynamics...")

  for x in tqdm(Xs) :
    i = 0
    dy = 1
    y = x.copy()
    ys = [y.copy()]

    while not stop(i,dy,y) :
      y += dyn(y)*alpha
      dy = np.linalg.norm(np.abs(y-ys[-1]))
      ys.append(y.copy())
      i += 1
    Ys.append(ys)
    # only save the converged points
    if i <= max_iter :
      x_arrived.append(y[0])
      y_arrived.append(y[1])
      if n == 3 :
        z_arrived.append(y[2])
    k = k+1

  for ys in Ys :
    ys_array = np.array(ys)
    plt.plot(ys_array[:,axx],ys_array[:,axy],color = "gray",alpha = 0.3)

  if arrows :
    print("plotting arrows...")
    for ys in tqdm(Ys):
        y = np.array(ys)
        for i in range((len(y) - 1)//factor):
          kara = y[i*factor]
          made = y[i*factor+1]-kara
          plt.arrow(kara[axx],kara[axy],made[axx],made[axy],head_width = 0.01,head_length = 0.03,fc = "black",ec = "black")

  x = Xs[:,axx]
  y = Xs[:,axy]

  plt.scatter(x,y,alpha = 0.5)


  if game_name == "prisonners_dilema" :
    plt.xlabel("cooperate")
    plt.ylabel("defect")
  else :
    plt.xlabel("X")
    plt.ylabel("Y")


  if arrived :
    if n == 3 :
      arrive = [x_arrived,y_arrived,z_arrived]
    elif n == 2 :
      arrive = [x_arrived,y_arrived]
    plt.scatter(arrive[axx],arrive[axy])

  plt.title(game_name)
  plt.grid(True)
  plt.show()

  # if n is 3, we plot a 3D plot of all arrived points
  if n == 3 :
    plt.subplot(131)
    plt.scatter(arrive[0],arrive[1])
    plt.xlabel("rock")
    plt.ylabel("scissors")

    plt.subplot(132)
    plt.scatter(arrive[0],arrive[2])
    plt.xlabel("rock")
    plt.ylabel("paper")

    plt.subplot(133)
    plt.scatter(arrive[1],arrive[2])
    plt.xlabel("scissors")
    plt.ylabel("paper")

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(arrive[0],arrive[1],arrive[2])

    plt.show()

    # time dependency plot 
    to_plot = np.array(Ys[len(Ys)//2])
    plt.plot(to_plot[:,0],label = "rock")
    plt.plot(to_plot[:,1],label = "scissors")
    plt.plot(to_plot[:,2],label = "paper")

    plt.xlabel("time")
    plt.legend()

    plt.show()



  return arrive

def main(_) :
    game,n = get_game(FLAGS.game_name)
    arr = FLAGS.arrows
    h = FLAGS.h
    alpha = FLAGS.alpha
    max_iter = FLAGS.max_iter
    max_tol = FLAGS.abs_tol
    arrived = True
    game_name = FLAGS.game_name
    factor = FLAGS.factor
    arrive = replicator_dynamics(game,h,max_iter,max_tol, alpha, arr, arrived, n,factor=factor,game_name = game_name,axis = (0,1))

if __name__ == "__main__":
   app.run(main)