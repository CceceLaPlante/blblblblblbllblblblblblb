import logging
import sys
from absl import app
from absl import flags
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms import random_agent

import torch
import matplotlib.pyplot as plt

import pyspiel
import numpy as np
from scipy.special import softmax

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_episodes", int(1000), "Number of train episodes.")
flags.DEFINE_boolean(
    "interactive_play", False,
    "Whether to run an interactive play with the agent after training.")
flags.DEFINE_integer("b_size", int(3), "size of the board")

flags.DEFINE_float("c", 1.4, "Exploration parameter in UCB")
flags.DEFINE_integer("max_sim", int(1e4), "Number of simulations")
flags.DEFINE_integer("print_losses", int(100), "Number of episodes between each print")
flags.DEFINE_integer("eval_against_random", int(100), "Number of episodes between each evaluation against random bot")

class evaluation_model() :
    def __init__ (self, input_shape,hidden_layers,lr) :
        torch.manual_seed(0)
        output_shape = 2
        self.model = torch.nn.Sequential()
        self.model.add_module("input",torch.nn.Linear(input_shape,hidden_layers[0]))
        for h in range(1,len(hidden_layers)) :
            self.model.add_module("hidden",torch.nn.Linear(hidden_layers[h-1],hidden_layers[h]))
            self.model.add_activation("activation",torch.nn.ReLU())

        self.model.add_module("output",torch.nn.Linear(hidden_layers[-1],output_shape))
        self.model.add_module("activation",torch.nn.Sigmoid())
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
    
    def forward(self,x) :
        return self.model(x)
    
    def train(self,x,y) :
        self.optimizer.zero_grad()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred,y)
        loss.backward()
        self.optimizer.step()

        return loss.item()


class DeepEval (mcts.Evaluator) :

    def __init__ (self,model) :
        self.model = model

    def evaluate(self, state) :
        with torch.no_grad() :
            s = state.observation_tensor().flatten()
            s = torch.tensor(s,dtype=torch.float32)
            return self.model.forward(s)
    
    def prior(self,state) :
        with torch.no_grad() :
            childs = [state.child(i) for i in state.legal_actions()]
            s = [c.observation_tensor().flatten() for c in childs]
            s = torch.tensor(s,dtype=torch.float32)
            answer =  softmax(self.model.forward(s))
            return {state.legal_actions()[i] : answer[i] for i in range(len(answer))}
    
    def learn (self, states, values) :
        losses = []
        for s,v in zip(states,values) :
            s = s.observation_tensor().flatten()
            s = torch.tensor(s,dtype=torch.float32)
            v = torch.tensor(v,dtype=torch.float32)
            l = self.model.train(s,v)
            losses.append(l)

        return np.array(losses)
    
def mbot_eval_against_random(g,mbot,random_bot,num_episodes) :
    sum_episode_rewards = np.zeros(2)
    bots = [mbot,random_bot]
    for _ in range(num_episodes) :
        s = g.new_initial_state()
        while not s.is_terminal() :
            bot = bots[s.current_player()]
            a = bot.step(s)
            s.apply_action(a)
        sum_episode_rewards += s.returns()
    return sum_episode_rewards / num_episodes
        

def deep_mcts() :
    c = FLAGS.c
    max_sim = FLAGS.max_sim
    num_ep = FLAGS.num_episodes
    print_losses = FLAGS.print_losses
    eval_against_random = FLAGS.eval_against_random
    alpha = 0.1
    game = "dots_and_boxes(num_rows={},num_cols={})".format(FLAGS.b_size,FLAGS.b_size)
    env = rl_environment.Environment(game)
    g = pyspiel.load_game(game)

    num_players = g.num_players()
    state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [64,64]
    m = evaluation_model(state_size,
                          hidden_layers_sizes,
                          0.01)
    

    
    deval = DeepEval(m)
    mbot = mcts.MCTSBot(g,
                        c,
                        max_sim,
                        deval)
    
    random_bot_eval = mcts.RandomRolloutEvaluator()
    random_bot = mcts.MCTSBot(g,
                                c,
                                max_sim,
                                random_bot_eval)
    every_episode_losses = []
    
    for e in range(num_ep) :
        state = g.new_initial_state()
        history = [state.clone()]
        values = []
        ep_len = 1
        while not state.is_terminal() :
            
            action = mbot.step(state)
            state.apply_action(action)

            history.append(state.clone())
            values.append([0,0])
            ep_len += 1

        values.append(state.returns())

        # time for backpropagation
        losses1, losses2 = None,None

        for i in range(ep_len-1) :
            k = ep_len - i-1
            values[k] = values[k+1]*alpha+values[k]*(1-alpha)
            l = deval.learn([history[k]],[values[k]])
            every_episode_losses.append(l)

        if e % print_losses == 0 :
            print("Episode : {} Losses : {} {}".format(e,losses1,losses2))

        if e % eval_against_random == 0 :
            r_mean = mbot_eval_against_random(env,mbot,random_bot,1000)
            print("Mean episode rewards: {}".format(r_mean))

    plt.plot(every_episode_losses)
            
        


        
        
if __name__ == "__main__":
    app.run(deep_mcts)
