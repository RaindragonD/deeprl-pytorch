import time
import os
from deeprl.trainer import Trainer
from deeprl.agents.pg_agent import PGAgent
from deeprl.agents.ac_agent import ACAgent
from deeprl.agents.dqn_agent import DQNAgent

def get_trainer(params):
        
        if (params['agent_name']=='pg'):
            params['agent_class'] = PGAgent
            params['train_batch_size'] = params['batch_size']
        elif (params['agent_name']=='a2c'):
            params['agent_class'] = ACAgent
            params['train_batch_size'] = params['batch_size']
        elif (params['agent_name']=='dqn'):
            params['agent_class'] = DQNAgent
        elif (params['agent_name']=='ddqn'):
            params['agent_class'] = DQNAgent
        else:
            print('Agent not implememnted. Terminating...')
            return

        agent_params = {
            'batch_size': params['batch_size'],
            'train_batch_size': params['train_batch_size'],
            'eval_batch_size': params['eval_batch_size'],

            'gamma': params['gamma'],
            'learning_rate': params['learning_rate'],
            'n_layers': params['n_layers'],
            'size': params['size'],
            
            'use_gpu': params['use_gpu'],
            'which_gpu': params['which_gpu'],

            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'num_actor_updates_per_agent_update': params['num_actor_updates_per_agent_update'],
            'num_target_updates': params['num_target_updates'],
            'num_grad_steps_per_target_update': params['num_grad_steps_per_target_update'],
            'target_update_freq': params['target_update_freq'],

            'double_q': params['agent_name']=='ddqn'
        }

        params['agent_params'] = agent_params

        # Initiate logger path
        data_path = './runs'
        logdir = params['exp_name'] or params['agent_name']+'_'+ params['env_name']+'_'+time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join(data_path, logdir)
        params['logdir'] = logdir
        if not(os.path.exists(logdir)):
            os.makedirs(logdir)
        else:
            for f in os.listdir(logdir):
                os.remove(os.path.join(logdir, f))
        print("LOGGING TO: ", logdir)
        
        return Trainer(params)

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--agent_name', type=str, default='ddqn')
    
    parser.add_argument('--epoch_size', type=int, default=200)
    parser.add_argument('--itr_per_epoch', type=int, default=10) # num of iterations per epoch for agent
    
    parser.add_argument('--num_critic_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--num_actor_updates_per_agent_update', type=int, default=1)
    parser.add_argument('--num_target_updates', '-ntu', type=int, default=100)
    parser.add_argument('--num_grad_steps_per_target_update', '-ngsptu', type=int, default=1)
    parser.add_argument('--target_update_freq', type=int, default=1) # dqn target update frequency

    parser.add_argument('--batch_size', '-b', type=int, default=1000) #steps collected per iteration
    parser.add_argument('--train_batch_size', '-tb', type=int, default=500) #steps collected per train iteration
    parser.add_argument('--eval_batch_size', '-eb', type=int, default=100) #steps collected per eval iteration
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--size', '-s', type=int, default=64)
    
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_gpu', '-gpu', action='store_true')
    parser.add_argument('--which_gpu', '-gpu_id', default=0)
    
    # parser.add_argument('--video_log_freq', type=int, default=-1)
    # parser.add_argument('--scalar_log_freq', type=int, default=1)
    # parser.add_argument('--save_params', action='store_true')

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    trainer = get_trainer(params)
    trainer.train()
    trainer.run_env()
    trainer.logger.close()

if __name__ == "__main__":
    main()
