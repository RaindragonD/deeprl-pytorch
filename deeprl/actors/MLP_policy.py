import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from .base_policy import BasePolicy
from deeprl.utils.pytorch_utils import FCNet

class MLPPolicy(BasePolicy, nn.Module):

    def __init__(self, ac_dim, ob_dim, 
        n_layers = 2, size = 64, dtype=torch.FloatTensor, 
        optimizer=optim.Adam, learning_rate=1e-4,
        training=True, discrete=True, **kwargs):
    
        super(MLPPolicy, self).__init__(**kwargs)

        # store params
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.dtype = dtype
        self.lr = learning_rate
        self.training = training
        self.discrete = discrete

        # init vars

        # create net for policy approximator
        self.policy = FCNet(np.prod(ob_dim), ac_dim, n_layers, size, output_activation=nn.Softmax).type(dtype)

        self.optimizer = optimizer(self.policy.parameters(), lr=self.lr)

    # def save(self, filepath):
    #     self.policy_saver.save(self.sess, filepath, write_meta_graph=False)

    # def restore(self, filepath):
    #     self.policy_saver.restore(self.sess, filepath)

    def update(self, obs_batch, actions, adv_vals):
        obs_batch = torch.from_numpy(obs_batch).type(self.dtype)
        adv_vals = torch.from_numpy(adv_vals).type(self.dtype)
        logits_n = self.policy(Variable(obs_batch.view(obs_batch.shape[0],-1)))

        # Calculate log(pi(a|s))
        log_probs = Variable(torch.Tensor()) 
        if self.discrete:
            for i in range(len(obs_batch)):

                action_distribution = Categorical(logits_n[i])
                action = torch.tensor(actions[i])
                log_prob = action_distribution.log_prob(action)
                if len(log_probs) > 0:
                    log_probs = torch.cat([log_probs, log_prob.reshape(1)])
                else:
                    log_probs = log_prob.reshape(1)
        else:
            pass

        loss = (torch.sum(torch.mul(log_probs, Variable(adv_vals)).mul(-1), -1))
        
        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_action(self, obs):

        with torch.no_grad():
            obs = torch.from_numpy(obs).type(self.dtype)
            action_probs = self.policy(obs.view(-1))

            if self.discrete:
                action_distribution = Categorical(action_probs)
                action = action_distribution.sample()
            else:
                pass

        return action.data.numpy()
