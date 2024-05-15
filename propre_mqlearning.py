import pyspiel
import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags
import sys
import os

from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python import rl_agent
from open_spiel.python.algorithms import tabular_qlearner
from scipy.special import softmax, expit
import random
import collections

from tqdm import tqdm

from open_spiel.python.egt.utils import game_payoffs_array

FLAGS = flags.FLAGS
flags.DEFINE_string("game_name","prisoners_dilemma","Game to play (prisoners_dilemma, subsidy_game, coordination_game, anti_coordination_game, battle_of_the_sex, biased_rps)")

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
    
class lenientB_qlearner (tabular_qlearner.QLearner) :
  def __init__(self,
               player_id,
               num_actions,
               start=None,
               step_size=0.1,
               epsilon_schedule=rl_tools.ConstantSchedule(0.2),
               discount_factor=1.0,
               centralized=False,
               temperature = 1):
    super().__init__(player_id, num_actions, step_size, epsilon_schedule, discount_factor, centralized)
    self.temperature = temperature
    if start :
      for i in range(len(start)) :
        self._q_values[str([0.0])][i] = start[i]


  # overwrite to change to the boltzmann distribution
  def _get_action_probs (self, info_state, legal_actions, epsilon) :
    probs = np.zeros(self._num_actions)
    q_values = [self._q_values[info_state][a] for a in legal_actions]
    #here it comes, da softmax (or boltzmann, or gibbs distribution)
    action = np.random.choice(range(len(legal_actions)), p = softmax(np.array(q_values)/self.temperature))
    probs = softmax(np.array(q_values)/self.temperature)
    
    return action, probs

  # to add the leniency, we will not train while taking decisions (evaluation mode)
  # and update q when one action have been taken k times

  def learn (self, state, action, reward,time_step, legal_actions) :

    # note : the state is the state BEFORE the action have been taken !
    #target = time_step.rewards[self._player_id]
    target = reward

    if not time_step.last():  # Q values are zero for terminal.
      target += self._discount_factor * max(
          [self._q_values[state][a] for a in legal_actions])

    prev_q_value = self._q_values[state][action]
    self._last_loss_value = target - prev_q_value
    self._q_values[state][action] += (
        self._step_size * self._last_loss_value)

    # Decay epsilon, if necessary.
    self._epsilon = self._epsilon_schedule.step()

    if time_step.last():  # prepare for the next episode.
      self._prev_info_state = None
      return

    self._prev_info_state = state
    self._prev_action = action

class lenientEPS_qlearner (tabular_qlearner.QLearner) :
  def __init__(self,
               player_id,
               num_actions,
               start=None,
               step_size=0.1,
               epsilon_schedule=rl_tools.ConstantSchedule(0.2),
               discount_factor=1.0,
               centralized=False):
    super().__init__(player_id, num_actions, step_size, epsilon_schedule, discount_factor, centralized)

  # to add the leniency, we will not train while taking decisions (evaluation mode)
  # and update q when one action have been taken k times

  def learn (self, state, action, reward,time_step, legal_actions) :

    # note : the state is the state BEFORE the action have been taken !
    #target = time_step.rewards[self._player_id]
    target = reward

    if not time_step.last():  # Q values are zero for terminal.
      target += self._discount_factor * max(
          [self._q_values[state][a] for a in legal_actions])

    prev_q_value = self._q_values[state][action]
    self._last_loss_value = target - prev_q_value
    self._q_values[state][action] += (
        self._step_size * self._last_loss_value)

    # Decay epsilon, if necessary.
    self._epsilon = self._epsilon_schedule.step()

    if time_step.last():  # prepare for the next episode.
      self._prev_info_state = None
      return

    self._prev_info_state = state
    self._prev_action = action

def learning (game, env,agents, k_leniency, nb_episodes,temperature) :
    mean_reward = np.array([0,0])
    mean_actions = np.array([0,0])
    num_actions = game.num_distinct_actions()


    lenient_queue1 = []
    lenient_queue2 = []

    action_count1 = np.zeros(num_actions)
    action_count2 = np.zeros(num_actions)

    s1 = np.zeros((num_actions))
    s2 = np.zeros((num_actions))

    sum_prob1 = np.zeros((num_actions))
    sum_prob2= np.zeros((num_actions))

    total_reward1 = 0
    total_reward2 = 0

    cumulatives1 = np.zeros((nb_episodes,num_actions))
    cumulatives2 = np.zeros((nb_episodes,num_actions))

    probs1 = np.zeros((nb_episodes,num_actions))
    probs2 = np.zeros((nb_episodes,num_actions))
    for cur_episode in range(nb_episodes):
        time_step = env.reset()

        # this is of course overkill for the use that we have but at least we can change
        # the game easely
        info_state1 = str(time_step.observations["info_state"][0])
        legal_actions1 = time_step.observations["legal_actions"][0]

        info_state2 = str(time_step.observations["info_state"][1])
        legal_actions2 = time_step.observations["legal_actions"][1]

        qval1 = np.array([agents[0]._q_values[info_state1][a] for a in legal_actions1])
        qval2 = np.array([agents[1]._q_values[info_state2][a] for a in legal_actions2])

        probs1[cur_episode,:] = softmax(qval1/temperature)
        probs2[cur_episode, :] = softmax(qval2/temperature)

        agent_output = agents[0].step(time_step,is_evaluation=True)
        agent_output2 = agents[1].step(time_step,is_evaluation=True)
        time_step = env.step([agent_output.action,agent_output2.action])

        reward1 = time_step.rewards[0]
        reward2 = time_step.rewards[1]

        total_reward1 += reward1
        total_reward2 += reward2

        action1 = agent_output.action
        action2 = agent_output2.action

        action_count1[action1] += 1
        action_count2[action2] += 1

        s1[action1] +=1
        s2[action2] +=1

        cumulatives1[cur_episode,:] = s1
        cumulatives2[cur_episode,:] = s2

        lenient_queue1.append( (info_state1, action1, reward1, time_step, legal_actions1) )
        lenient_queue2.append( (info_state2, action2, reward2, time_step, legal_actions2) )

        max_actions1 = np.max(action_count1), np.argmax(action_count1)
        max_actions2 = np.max(action_count2), np.argmax(action_count2)

        sum_prob1 += agent_output.probs
        sum_prob2 += agent_output2.probs

        if max_actions1[0] == k_leniency :
            max_action_reward = float("-inf")
            # optimistic version
            for (s, a, r, ts, la) in lenient_queue1 :
               if a == max_actions1[1] :
                  if r > max_action_reward :
                      max_action_reward = r
            
            for (s, a, r, ts, la) in lenient_queue1 :
                new_lq = []
                if a == max_actions1[1] and r == max_action_reward:
                    agents[0].learn(s,a,r,ts,la)
                elif a != max_actions1[1] :
                    new_lq.append((s, a, r, ts, la))

            action_count1[max_actions1[1]] = 0
            lenient_queue2 = new_lq.copy()

        if max_actions2[0] == k_leniency :
            max_action_reward = float("-inf")
            # optimistic version
            for (s, a, r, ts, la) in lenient_queue1 :
               if a == max_actions2[1] :
                  if r > max_action_reward :
                      max_action_reward = r

            for (s, a, r, ts, la) in lenient_queue2 :
                new_lq = []
                if a == max_actions2[1] and r == max_action_reward:
                    agents[1].learn(s,a,r,ts,la)
                elif a != max_actions2[1] :
                    new_lq.append((s, a, r, ts, la))

            lenient_queue2 = new_lq.copy()
            action_count2[max_actions2[1]] = 0

    return cumulatives1, cumulatives2, sum_prob1, sum_prob2,probs1,probs2,total_reward1, total_reward2

def main(_) :
    game,n = get_game(FLAGS.game_name)
    env = rl_environment.Environment(game)
    num_players = env.num_players
    num_actions = env.action_spec()["num_actions"]

    discount_factor = 0.1
    k_leniency = 10
    eps = 0.2
    step_size = 0.1
    nb_episodes = 1000

    temperature = 0.5
    h = 10
    m = 1

    if n == 2 :
        Xs1 = np.arange(0,m,1/h)
        np.random.shuffle(Xs1)

        Xs2 = np.arange(0,m,1/h)
        np.random.shuffle(Xs2)

        print(Xs1)
        Xs1 = np.array([m-Xs1.copy(),Xs1]).T
        Xs2 = np.array([m-Xs2.copy(),Xs2]).T
        print(Xs1.shape)

        Xs = [Xs1,Xs2]
    print("training agents couples")
    for idx in tqdm(range(len(Xs[0]))) :
        start = (list(Xs[0][idx]),list(Xs[1][idx]))
        agents = [
        lenientB_qlearner(player_id=idx,
                        num_actions=num_actions,
                        discount_factor = discount_factor,
                        step_size = step_size,
                        epsilon_schedule=rl_tools.ConstantSchedule(eps),
                        temperature = temperature,
                        start = start[idx])
        for idx in range(num_players)

        ]

        cumulatives1, cumulatives2, sum_prob1, sum_prob2,probs1,probs2,total_reward1, total_reward2 = learning(game, env,agents, k_leniency, nb_episodes,temperature)
        plt.subplot(121)
        plt.plot(probs1[:,0])
        plt.subplot(122)
        plt.plot(np.abs(probs1[:,0]-probs2[:,0]))

    plt.show()

if __name__ == "__main__" :
   app.run(main)


