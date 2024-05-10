import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.checkpoints import get_checkpoint_info
from ray.tune.registry import register_env
import matplotlib.pyplot as plt
import torch
import pickle
import os

from logger import Logger

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')
    
# print(f'Device: {device}')

def preprocess_env(env):
    env = gym.wrappers.AtariPreprocessing(env, noop_max = 30, frame_skip = 4, screen_size = 84, 
                                                  terminal_on_life_loss = False, grayscale_obs = True, 
                                                  grayscale_newaxis = False, scale_obs = False)
    
    env = gym.wrappers.FrameStack(env, 4)

    return env

def create_env():

    return gym.make("BreakoutNoFrameskip-v4")

def load_algo(algo_dir):
    print('Loading algo...')
    dir = os.path.join(os.getcwd(), algo_dir)
    print(f'Loading algo from: {dir}')
    with open(f'{dir}.pkl', 'rb') as file:
        algo = pickle.load(file)

    return algo

# config = (
#     ppo.PPOConfig()
#     .environment("BreakoutNoFrameskip-v4", render_env=True)
#     .env_runners(num_env_runners=2)
#     .framework("torch")
#     .training(model={"fcnet_hiddens": [64, 64], "dim": 84})
#     .evaluation(evaluation_num_env_runners=1)
#     .resources(num_cpus_per_worker=1, num_gpus_per_worker=0.1)
# )

# config = ppo.PPOConfig().environment("BreakoutNoFrameskip-v4", render_env=True, is_atari=True).env_runners(num_env_runners=2).framework("torch").training(model={"fcnet_hiddens": [64, 64]}).evaluation(evaluation_num_env_runners=1).resources(num_cpus_per_worker=1, num_gpus_per_worker=0.1)

wandb_logger = Logger(f"inm707_breakout_test_100_steps", project='INM705_CW')
logger = wandb_logger.get_logger()

config = ppo.PPOConfig().environment("BreakoutNoFrameskip-v4").training(model={"fcnet_hiddens": [3], 
                                                                               "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]], 
                                                                               "framestack": True, 
                                                                               "dim": 42, 
                                                                               "grayscale": False
                                                                               }).env_runners(num_env_runners=2).framework("torch").resources(num_cpus_per_worker=1, num_gpus_per_worker=0.1).evaluation(evaluation_num_env_runners=1)

algo = config.build()

for _ in range(100):

    print(f'\n\tStep: {_}\n')
    print(algo.train())  # 3. train it,

# algo.train()
# save_result = algo.save("./checkpoints")
# print(save_result)

# print('Saving agent...')
# with open(f'checkpoints/ppo_test_2.pkl', 'wb') as file:
#     pickle.dump(algo, file)

print('Saving weights...')
with open(f'checkpoints/ppo_test_weights.pkl', 'wb') as file:
    pickle.dump(algo.get_policy().get_weights(), file)

# algo.evaluate()  # 4. and evaluate it.

# env = gym.make("FrozenLake-v1")

# env = preprocess_env(gym.make("BreakoutNoFrameskip-v4", render_mode="human"))

########################################## Testing ###############################################

# env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
# obs, info = env.reset()

# algo = config.build()
# algo = Algorithm.from_checkpoint('./checkpoints\\algorithm_state.pkl')

# algo = config.build()
# algo.restore('./checkpoints/algorithm_state.pkl')

# my_restored_policy = Policy.from_checkpoint("./checkpoints/policies/default_policy/policy_state.pkl", policy_ids=['default_policy'])

# algo = config.build()

# print('Loading checkpoint')
# with open('./checkpoints/ppo_test_weights.pkl', 'rb') as file:
#     weights = pickle.load(file)
#     algo.get_policy().set_weights(weights)

# num_episodes = 0
# episode_reward = 0.0

# algo.evaluate()

# while num_episodes < 10:
#     # Compute an action (`a`).
#     a = algo.compute_single_action(
#         observation=obs,
#         explore="store_true",
#         policy_id="default_policy",  # <- default value
#     )
#     # Send the computed action `a` to the env.
#     obs, reward, done, truncated, _ = env.step(a)
#     episode_reward += reward
#     # Is the episode `done`? -> Reset.
#     if done:
#         print(f"Episode {num_episodes} done: Total reward = {episode_reward}")
#         obs, info = env.reset()
#         num_episodes += 1
#         episode_reward = 0.0

algo.stop()

ray.shutdown()

#########################################################################################################

# ray.init()
# register_env("BreakoutNoFrameskip-v4", lambda config: create_env())

# algo = ppo.PPO(env="BreakoutNoFrameskip-v4")
# agent = ppo.PPOTrainer(env=env)

# env.reset()
# n_actions = env.action_space.n
# state_dim = env.observation_space.shape

# for _ in range(50):
#     print(_)
#     # print(algo.train())
#     obs = env.step(env.action_space.sample())