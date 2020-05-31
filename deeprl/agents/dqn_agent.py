import numpy as np 

from .base_agent import BaseAgent
from deeprl.actors.argmax_policy import ArgMaxPolicy
from deeprl.critics.dqn_critic import DQNCritic
from deeprl.utils.replay_buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(DQNAgent, self).__init__()

        # init vars
        self.env = env 
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.target_update_freq = self.agent_params['target_update_freq']

        # actor and critic
        self.critic = DQNCritic(self.agent_params)
        self.actor = ArgMaxPolicy(self.critic, self.agent_params['dtype'])
        self.num_param_updates = 0

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size)

    def train(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):
        if (self.num_param_updates % self.target_update_freq == 0):
            self.critic.update_target()
        self.num_param_updates += 1;
        loss = self.critic.update(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
        return loss

    def save_model(self, path):
        self.critic.save_model(path)