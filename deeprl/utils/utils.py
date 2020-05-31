import numpy as np

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : None,
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

def sample_trajectory(env, policy, max_path_length, render=False):

    # initialize env for the beginning of a new rollout
    ob = env.reset() # HINT: should be the output of resetting the env

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # render image of the simulated env
        if render:
            env.render()

        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        rollout_done = done or steps==max_path_length
        terminals.append(rollout_done)
        
        if rollout_done: 
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals), steps

def sample_trajectories(env, policy, min_batch_size, max_path_len):

    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_batch_size:
        path, steps = sample_trajectory(env, policy, max_path_len)
        paths.append(path)
        timesteps_this_batch += steps

    return paths, timesteps_this_batch