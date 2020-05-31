import numpy as np 

from .base_agent import BaseAgent
from deeprl.actors.MLP_policy import MLPPolicy
from deeprl.utils.replay_buffer import ReplayBuffer

class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

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
            dtype = self.agent_params['dtype'])
        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size)

    def train(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):
        q_values = self.calculate_q_vals(re_batch)
        advantage_values = self.estimate_advantage(q_values)
        loss = self.actor.update(ob_batch, ac_batch, advantage_values)
        return loss

    def calculate_q_vals(self, rews_list):
        # calculate reward_to_go q-values
        q_values = np.concatenate([self.calc_reward_to_go(r) for r in rews_list])
        return q_values

    def calc_reward_to_go(self, rewards):
        """
            Input:
                a list of length T 
                a list of rewards {r_0, r_1, ..., r_t', ... r_{T-1}} from a single rollout of length T
            Output: 
                a list of length T
                a list where the entry in each index t is sum_{t'=t}^{T-1} gamma^(t'-t) * r_{t'}
        """

        all_discounted_cumsums = []
        for start_time_index in range(rewards.shape[0]): 
            indices = np.arange(start_time_index, len(rewards))
            discounts = np.power(self.gamma, indices - start_time_index)
            discounted_rtg = discounts * rewards[indices]
            sum_discounted_rtg = np.sum(discounted_rtg)
            all_discounted_cumsums.append(sum_discounted_rtg)

        list_of_discounted_cumsums = np.array(all_discounted_cumsums)
        return list_of_discounted_cumsums 

    def estimate_advantage(self, q_values):
        # mean baseline
        advs = (q_values - q_values.mean())
        # standardize advantages
        advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-8)

        return advs
    