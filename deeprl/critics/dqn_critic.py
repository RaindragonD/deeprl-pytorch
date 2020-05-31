import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from .base_critic import BaseCritic
from deeprl.utils.pytorch_utils import FCNet, CovNet

class DQNCritic(BaseCritic):
    def __init__(self, params, optimizer=optim.Adam):
        super(DQNCritic, self).__init__()

        self.ob_dim = params['ob_dim']
        self.ac_dim = params['ac_dim']
        self.discrete = params['discrete']
        self.size = params['size']
        self.n_layers = params['n_layers']
        self.lr = params['learning_rate']
        self.num_target_updates = params['num_target_updates']
        self.num_grad_steps_per_target_update = params['num_grad_steps_per_target_update']
        self.gamma = params['gamma']
        self.double_q = params['double_q']
        self.dtype = params['dtype']

        # self.q_func = FCNet(np.prod(self.ob_dim), self.ac_dim, self.n_layers, self.size, output_activation = None).type(self.dtype)
        # self.target_q_func = FCNet(np.prod(self.ob_dim), self.ac_dim, self.n_layers, self.size, output_activation = None).type(self.dtype)
        
        self.q_func = CovNet(self.ob_dim, self.ac_dim).type(self.dtype)
        self.target_q_func = CovNet(self.ob_dim, self.ac_dim).type(self.dtype)
        self.target_q_func.load_state_dict(self.q_func.state_dict())
        self.target_q_func.eval()
        self.optimizer = optimizer(self.q_func.parameters(), lr=self.lr)
        self.lossCrit = F.smooth_l1_loss

    def update(self, ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch):
        # ob_batch = torch.from_numpy(ob_batch).type(self.dtype)
        # next_ob_batch = torch.from_numpy(next_ob_batch).type(self.dtype)
        ob_batch = torch.from_numpy(ob_batch).type(self.dtype).permute(0,3,1,2)
        next_ob_batch = torch.from_numpy(next_ob_batch).type(self.dtype).permute(0,3,1,2)
        ac_batch = torch.from_numpy(ac_batch).type(torch.int64).view(-1,1)

        predicted_vals = self.q_func(Variable(ob_batch)).gather(1, ac_batch)
        next_state_qvals = self.target_q_func(next_ob_batch)

        if self.double_q:
            target_actions = self.q_func(next_ob_batch).argmax(1).view(-1,1)
            next_state_vals = next_state_qvals.gather(1, target_actions).view(1,-1).detach().numpy()
        else:
            next_state_vals = next_state_qvals.max(1)[0].detach().numpy()
        target_vals = re_batch + self.gamma * next_state_vals * (1-terminal_batch)
        target_vals = torch.from_numpy(target_vals).type(self.dtype).view(-1,1).detach()
        loss = self.lossCrit(predicted_vals, target_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_qval(self, obs):
        obs = torch.from_numpy(obs).type(self.dtype)
        # return self.q_func(Variable(obs.view(obs.shape[0],-1)))   
        return self.q_func(Variable(obs.permute(0,3,1,2)))   

    def update_target(self):
        self.target_q_func.load_state_dict(self.q_func.state_dict())

    def save_model(self, path):
        torch.save(self.q_func.state_dict(), path)