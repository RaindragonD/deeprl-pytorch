import numpy as np 
from collections import OrderedDict

from .base_agent import BaseAgent
from deeprl.actors.MLP_policy import MLPPolicy
from deeprl.critics.bootstrapped_critic import BootstrappedCritic
from deeprl.utils.replay_buffer import ReplayBuffer

class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        # init vars
        self.env = env 
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']

        # actor/policy
        self.actor = MLPPolicy(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            dtype=self.agent_params['dtype']) 
        # replay buffer
        self.critic = BootstrappedCritic(self.agent_params)
        self.replay_buffer = ReplayBuffer(1000000)

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=True)

    def train(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):
        for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            cr_loss = self.critic.update(ob_batch, next_ob_batch, re_batch, terminal_batch)
        adv_vals = self.estimate_advantage(ob_batch, next_ob_batch, re_batch, terminal_batch)
        for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
            ac_loss = self.actor.update(ob_batch, ac_batch, adv_vals)

        loss = OrderedDict()
        loss['Critic_Loss'] = cr_loss
        loss['Actor_Loss'] = ac_loss
        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        v_s = self.critic.get_value(ob_no)
        v_sp = self.critic.get_value(next_ob_no)
        advs = re_n + self.gamma * v_sp.detach().numpy() * (1-terminal_n) - v_s.detach().numpy()
        # standardize advantages
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)

        return advs