import time
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

import gym
from gym.wrappers import Monitor
import torch
from torch.utils.tensorboard import SummaryWriter
from deeprl.utils.utils import *

class Trainer(object):
    def __init__(self, params):

        # Get params, initiate bookkeeping vars, logger
        self.params = params
        self.start_time = time.time()
        self.total_envsteps = 0
        self.average_return = []
        self.std_return = []
        self.logger = SummaryWriter(self.params['logdir'])
        
        # Make the environment
        self.env = gym.make(self.params['env_name'])

        # Set random seeds
        seed = self.params['seed']
        self.env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Maximum length for episodes
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps

        # Set agent env dependent parameters
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.params['agent_params']['discrete'] = discrete
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = self.env.observation_space.shape

        # gpu
        self.params['agent_params']['dtype'] = torch.cuda.FloatTensor if self.params['use_gpu'] else torch.FloatTensor
        self.params['agent_params']['which_gpu'] = self.params['which_gpu']

        # Create the agent
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def train(self):

        # init vars at beginning of training
        # self.total_envsteps = 0
        # self.start_time = time.time()

        # start training
        for epoch in range(self.params['epoch_size']):

            # sample trajectories
            paths, steps = sample_trajectories(self.env, self.agent.actor, self.params['batch_size'], self.params['ep_len'])
            self.total_envsteps += steps
            self.agent.add_to_replay_buffer(paths)

            for itr in range(self.params['itr_per_epoch']):
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
                loss = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)

            self.perform_logging(epoch, paths, loss)
        
        averages = np.array(self.average_return)
        stds = np.array(self.std_return)
        fig, ax = plt.subplots(1,1);
        ax.plot(averages)
        ax.fill_between(range(self.params['epoch_size']), averages-stds, averages+stds, color='orange', alpha=0.2)
        ax.set_title('reward')
        ax.set_xlabel('epoch'); ax.set_ylabel('reward')
        self.logger.add_figure('reward history', fig)

        self.agent.save_model('./model')

    def run_env(self):
        # self.env = gym.wrappers.Monitor(self.env, "./vid", video_callable=lambda episode_id: True,force=True)
        ob = self.env.reset()
        steps = 0
        while True:
            self.env.render()
            ac = self.agent.actor.get_action(ob)
            ob, rew, done, _ = self.env.step(ac)
            steps += 1
            rollout_done = done or steps==self.params['ep_len']
            if rollout_done: 
                break
        self.env.close()
        print("final step number: ", steps)

    def perform_logging(self, epoch, paths, loss):

        # returns, for logging
        train_returns = [path["reward"].sum() for path in paths]

        # episode lengths, for logging
        train_ep_lens = [len(path["reward"]) for path in paths]

        # decide what to log
        logs = OrderedDict()
        logs["AverageReturn"] = np.mean(train_returns)
        self.average_return.append(logs["AverageReturn"])
        logs["StdReturn"] = np.std(train_returns)
        self.std_return.append(logs["StdReturn"])
        logs["MaxReturn"] = np.max(train_returns)
        logs["MinReturn"] = np.min(train_returns)
        logs["AverageEpLen"] = np.mean(train_ep_lens)

        logs["EnvstepsSoFar"] = self.total_envsteps
        logs["TimeSinceStart"] = time.time() - self.start_time
        if isinstance(loss, dict):
            logs.update(loss)
        else:
            logs["Training loss"] = loss

        if epoch == 0:
            self.initial_return = np.mean(train_returns)
        logs["Initial_DataCollection_AverageReturn"] = self.initial_return

        # perform the logging
        if (epoch+1) % 20 == 0:
            print('epoch {}: {}'.format(epoch, logs["AverageReturn"]))
        for key, value in logs.items():
            self.logger.add_scalar(key, value, epoch)
        self.logger.flush()
        
        