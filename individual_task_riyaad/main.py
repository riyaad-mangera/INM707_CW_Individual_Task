import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
import pickle
import os
import numpy as np
from logger import Logger

def preprocess_env(env):
    env = gym.wrappers.AtariPreprocessing(env, noop_max = 30, frame_skip = 4, screen_size = 84, 
                                                  terminal_on_life_loss = False, grayscale_obs = True, 
                                                  grayscale_newaxis = False, scale_obs = False)
    
    env = gym.wrappers.FrameStack(env, 4)

    return env

def save_checkpoint(dir, weights):
    print('Saving weights...')
    with open(dir, 'wb') as file:
        pickle.dump(weights, file)

def load_checkpoint(dir):
    print('Loading checkpoint')
    with open(dir, 'rb') as file:
        weights = pickle.load(file)
        print('Checkpoint loaded')

        return weights

def create_config(env):
    config = ppo.PPOConfig()
    config.environment(env, 
                       env_config={"frameskip": 1,
                                   "repeat_action_probability": 0.0,
                                   }, clip_rewards=True,)

    config.training(model={"fcnet_hiddens": [256], 
                        "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [64, [4, 4], 2], [128, [11, 11], 1]], 
                        "framestack": True, 
                        "dim": 84, 
                        "grayscale": False,
                        })
    
    config.env_runners(num_env_runners=2)
    config.framework("torch")
    config.resources(num_cpus_per_worker=1, num_gpus_per_worker=0.1)
    config.evaluation(evaluation_num_env_runners=1)

    config.lambda_ = 0.95
    config.kl_coeff = 0.5
    config.clip_rewards = True
    config.clip_param = 0.1
    config.vf_clip_param = 10.0
    config.entropy_coeff = 0.01
    config.train_batch_size = 5000
    config.rollout_fragment_length = 100
    config.sgd_minibatch_size = 500
    config.num_sgd_iter = 10

    return config

def train(algo, logger, steps = 1, checkpoint_dir = ''):
    
    if checkpoint_dir != '':
        checkpoint_weights = load_checkpoint(checkpoint_dir)
        algo.get_policy().set_weights(checkpoint_weights)

    for step in range(steps):

        print(f'\n\tStep: {step}\n')
        result = algo.train()

        print(result)

        if logger != '':
            logger.log({'episode_return_mean': result['episode_return_mean']})

        if step % 10 == 0:
            save_checkpoint(f'checkpoints/ppo_test_weights_step_{step}.pkl', algo.get_policy().get_weights())

    save_checkpoint(f'checkpoints/ppo_test_weights_final.pkl', algo.get_policy().get_weights())

    return algo

def test(algo, checkpoint_dir):

    checkpoint_weights = load_checkpoint(checkpoint_dir)
    algo.get_policy().set_weights(checkpoint_weights)

    env = gym.make(ENV_NAME, render_mode="human")
    obs, info = env.reset()

    num_episodes = 0
    episode_reward = 0.0

    algo.evaluate()

    state = init_state = [np.zeros(84, np.float32)]

    while num_episodes < 10:
        
        action, state, _ = algo.compute_single_action(
            observation = obs,
            state = state,
            explore = "store_true",
            policy_id = "default_policy"
        )
        
        obs, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        
        if done:
            print(f"Episode {num_episodes}: Total reward = {episode_reward}")
            obs, info = env.reset()
            num_episodes += 1
            episode_reward = 0.0
            state = init_state

ENV_NAME = "BreakoutNoFrameskip-v4"

logger = ''
# wandb_logger = Logger(f"inm707_breakout_test", project='INM707_CW')
# logger = wandb_logger.get_logger()

config = create_config(env = ENV_NAME)
algo = config.build()

algo = train(algo, logger, steps = 10, checkpoint_dir = '')

# test(algo, checkpoint_dir = './checkpoints/ppo_algo_weights.pkl')

algo.stop()
ray.shutdown()