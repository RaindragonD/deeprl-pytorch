import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from .base_critic import BaseCritic
from deeprl.utils.pytorch_utils import FCNet

class BootstrappedCritic(BaseCritic):
    def __init__(self, params, optimizer=optim.Adam):
        super(BootstrappedCritic, self).__init__()

        self.ob_dim = params['ob_dim']
        self.ac_dim = params['ac_dim']
        self.discrete = params['discrete']
        self.size = params['size']
        self.n_layers = params['n_layers']
        self.lr = params['learning_rate']
        self.num_target_updates = params['num_target_updates']
        self.num_grad_steps_per_target_update = params['num_grad_steps_per_target_update']
        self.gamma = params['gamma']
        self.dtype = params['dtype']

        self.value_func = FCNet(np.prod(self.ob_dim), 1, self.n_layers, self.size, output_activation = None).type(self.dtype)
        self.optimizer = optimizer(self.value_func.parameters(), lr=self.lr)
        self.lossCrit = nn.MSELoss()

    def update(self, ob_no, next_ob_no, re_n, terminal_n):
        ob_no = torch.from_numpy(ob_no).type(self.dtype)
        next_ob_no = torch.from_numpy(next_ob_no).type(self.dtype)

        for i in range(self.num_grad_steps_per_target_update * self.num_target_updates):
            if i % self.num_grad_steps_per_target_update == 0:
                targ_val = re_n + self.gamma * self.value_func(next_ob_no).detach().numpy() * (1-terminal_n)
                targ_val = torch.from_numpy(targ_val).type(self.dtype)
            loss = self.lossCrit(targ_val, self.value_func(ob_no))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def get_value(self, obs):
        obs = torch.from_numpy(obs).type(self.dtype)
        return self.value_func(Variable(obs.view(obs.shape[0],-1)))    