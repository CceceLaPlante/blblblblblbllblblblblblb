import logging
import sys
from absl import app
from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import dqn
from open_spiel.python.algorithms import random_agent

import pyspiel

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(2000), "Number of train episodes.")
flags.DEFINE_boolean(
    "interactive_play", True,
    "Whether to run an interactive play with the agent after training.")
flags.DEFINE_integer("b_size", int(3), "size of the board")
flags.DEFINE_integer("freez" , int(10),"number of episodes each agents are frozen when the other is training")

def command_line_action(time_step):
  """Gets a valid action from the user on the command line."""
  current_player = time_step.observations["current_player"]
  legal_actions = time_step.observations["legal_actions"][current_player]
  action = -1

  while action not in legal_actions:
    print("Choose an action from {}:".format(legal_actions))
    sys.stdout.flush()
    action_str = input()
    try:
      action = int(action_str)
    except ValueError:
      continue

  return action

def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(trained_agents)
  sum_episode_rewards = np.zeros(num_players)

  for player_pos in range(num_players):
    cur_agents = random_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]

    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0

      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards

  return sum_episode_rewards / num_episodes

def main(_): 
    b_size = FLAGS.b_size
    game = "dots_and_boxes(num_rows={},num_cols={})".format(b_size,b_size)
    env = rl_environment.Environment(game)
    g = pyspiel.load_game(game)
    human_player = 1

    num_players = g.num_players()
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [128, 128]
    replay_buffer_capacity = int(1e5)
    train_episodes = FLAGS.num_episodes
    loss_report_interval = 1000
    
    with tf.Session() as sess:
        dqn_agent1 = dqn.DQN(
            sess,
            player_id=0,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity)
        dqn_agent2 = dqn.DQN(
            sess,
            player_id=1,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=replay_buffer_capacity)

        agents = [dqn_agent1, dqn_agent2]

        sess.run(tf.global_variables_initializer())
        random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]

        for ep in tqdm(range(train_episodes)) :
          # regulary show the loss
            if ep and ep % loss_report_interval == 0 :

                logging.info("[%s/%s] DQN loss: %s", ep, train_episodes, agents[0].loss)
                r_mean = eval_against_random_bots(env, agents, random_agents, 100)
                logging.info("Mean episode rewards: %s", r_mean)
            time_step = env.reset()



            while not time_step.last() :
                player_id = time_step.observations["current_player"]
                # compute the action to be player, and play it
                agent_output = agents[player_id].step(time_step)
                time_step = env.step([agent_output.action])
        
            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)
            
        # Evaluate against random agent
        random_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]
        
        r_mean = eval_against_random_bots(env, agents, random_agents, 1000)
        logging.info("Mean episode rewards: %s", r_mean)

        if not FLAGS.interactive_play:
            return
        
        # Play from the command line against the trained DQN agent.
        while True:
            logging.info("You are playing as %s", "X" if human_player else "0")
            time_step = env.reset()
            st = g.new_initial_state()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_id == human_player:
                    agent_out = agents[human_player].step(time_step, is_evaluation=True)
                    print(st)
                    action = command_line_action(time_step)
                else:
                    agent_out = agents[1 - human_player].step(
                        time_step, is_evaluation=True)
                    action = agent_out.action
                time_step = env.step([action])
                st.apply_action(action)

            print(st)

            logging.info("End of game!")
            if time_step.rewards[human_player] > 0:
                logging.info("You win")
            elif time_step.rewards[human_player] < 0:
                logging.info("You lose")
            else:
                logging.info("Draw")
            # Switch order of players
            human_player = 1 - human_player


if __name__ == "__main__":
  app.run(main)
