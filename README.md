Sample run
```sh
pip install -r requirements.txt
pip install -e .
cd deeprl
python3 main.py -env_name CartPole-v0 --agent_name PG
```


Arguments:
```sh
-h, --help
--env_name, gym environment name
--exp_name, experiment name (for tensorboard logging directory name)
--agent_name, agent name, supported arguments are pg (policy gradient), a2c (Advantage Actor-Critic), and dqn (Deep Q-Networks)
--epoch_size
--itr_per_epoch
--num_critic_updates_per_agent_update
--num_actor_updates_per_agent_update
--num_target_updates, -ntu
--num_grad_steps_per_target_update, -ngsptu
--target_update_freq
--batch_size, -b
--train_batch_size, -tb
--eval_batch_size, -eb
--ep_len
--gamma, discount factor
--learning_rate, -lr
--n_layers, -l
--size, -s, size of fully-connected layers
--seed
--use_gpu, -gpu
--which_gpu, -gpu_id
```