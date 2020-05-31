import torch

class ArgMaxPolicy(object):

    def __init__(self, critic, dtype=torch.FloatTensor):
        self.critic = critic
        self.dtype = dtype

    def get_action(self, obs):

        with torch.no_grad():
            obs = torch.from_numpy(obs).type(self.dtype)
            qvals = self.critic.q_func(obs.unsqueeze(0).permute(0,3,1,2))
            action = torch.argmax(qvals)
            return action.data.numpy()