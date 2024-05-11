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
import numpy as np
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
import warnings
warnings.filterwarnings("ignore")

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
#     .environment("BreakoutNoFrameskip-v4")
#     .env_runners(num_env_runners=2)
#     .framework("torch")
#     .training(model={"fcnet_hiddens": [64, 64], "dim": 84})
#     .evaluation(evaluation_num_env_runners=1)
#     .resources(num_cpus_per_worker=1, num_gpus_per_worker=0.1)
# )

# config = (ppo.PPOConfig()
#           .environment("BreakoutNoFrameskip-v4")
#           .training(model={"fcnet_hiddens": [3], 
#                            "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [64, [4, 4], 2], [128, [11, 11], 1]], 
#                            "framestack": True, 
#                            "dim": 84, 
#                            "grayscale": False,
#                            })
#             .env_runners(num_env_runners=2)
#             .framework("torch")
#             .resources(num_cpus_per_worker=1, num_gpus_per_worker=0.1)
#             .evaluation(evaluation_num_env_runners=1)
#           )

# config = ppo.PPOConfig().environment("BreakoutNoFrameskip-v4").training(model={"fcnet_hiddens": [3], 
#                                                                                "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [64, [4, 4], 2], [128, [11, 11], 1]], 
#                                                                                "framestack": True, 
#                                                                                "dim": 84, 
#                                                                                "grayscale": False,
#                                                                                }).env_runners(num_env_runners=2).framework("torch").resources(num_cpus_per_worker=1, num_gpus_per_worker=0.1).evaluation(evaluation_num_env_runners=1)
        #  .lambda(0.95)
        # .kl_coeff(0.5)
        # .clip_rewards(True)
        # .clip_param(0.1)
        # .vf_clip_param(10.0)
        # .entropy_coeff(0.01)
        # .train_batch_size(5000)
        # .rollout_fragment_length(100)
        # .sgd_minibatch_size(500)
        # .num_sgd_iter(10)

wandb_logger = Logger(f"inm707_breakout_test", project='INM707_CW')
logger = wandb_logger.get_logger()

config = ppo.PPOConfig()
config.environment("BreakoutNoFrameskip-v4", 
                   env_config={"frameskip": 1,
                               "full_action_space": False,
                               "repeat_action_probability": 0.0,
                               }, clip_rewards=True,)

config.training(model={"fcnet_hiddens": [256], 
                       "conv_filters": [[16, [4, 4], 2], [32, [4, 4], 2], [64, [4, 4], 2], [128, [11, 11], 1]], #[[64, [5, 5], 2], [128, [5, 5], 2], [128, [5, 5], 2], [256, [5, 5], 2], [256, [6, 6], 1]], #[[16, [4, 4], 2], [32, [4, 4], 2], [64, [4, 4], 2], [128, [11, 11], 1]], 
                       "framestack": True, 
                       "dim": 84, 
                       "grayscale": False,
                       })
# config.training(
#         model={
#             "vf_share_layers": True,
#             "conv_filters": [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]],
#             "framestack": True,
#             "conv_activation": "relu",
#             "post_fcnet_hiddens": [256],
#         })
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
# config.env_config = {"frameskip": 1, "full_action_space": False, "repeat_action_probability": 0.0}

algo = config.build()

print('Loading checkpoint')
with open('./checkpoints/fifth_test/ppo_test_weights_step_1.pkl', 'rb') as file:
    weights = pickle.load(file)
    algo.get_policy().set_weights(weights)

print('Checkpoint loaded')

for step in range(1000): #range(50, 100):

    print(f'\n\tStep: {step}\n')
    result = algo.train()

    print(result)

    # if step % 10 == 0:
    print('Saving weights...')
    with open(f'checkpoints/ppo_test_weights_step_{step}.pkl', 'wb') as file:
        pickle.dump(algo.get_policy().get_weights(), file)

print('Saving weights...')
with open(f'checkpoints/ppo_test_weights_final.pkl', 'wb') as file:
    pickle.dump(algo.get_policy().get_weights(), file)

####################################################################################################

# print('Saving weights...')
# with open(f'checkpoints/ppo_test_weights_final.pkl', 'wb') as file:
#     pickle.dump(algo.get_policy().get_weights(), file)

# algo.train()
# save_result = algo.save("./checkpoints")
# print(save_result)

# print('Saving agent...')
# with open(f'checkpoints/ppo_test_2.pkl', 'wb') as file:
#     pickle.dump(algo, file)

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

########################################## RUN BELOW ####################################################

algo = config.build()
# print(algo.get_policy().get_initial_state())
# print(f'\n\n{config["model"]}\n\n')

# print('Loading checkpoint')
# with open('./checkpoints/ppo_test_weights_step_1.pkl', 'rb') as file:
#     weights = pickle.load(file)
#     algo.get_policy().set_weights(weights)

# print('Checkpoint loaded')

env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
# env = wrap_deepmind(env)
obs, info = env.reset()

num_episodes = 0
episode_reward = 0.0

algo.evaluate()

state = init_state = [np.zeros(84, np.float32)]

while num_episodes < 10:
    # Compute an action (`a`).
    a, state, _ = algo.compute_single_action(
        observation=obs,
        state=state,
        # explore=False, #"store_true",
        policy_id="default_policy",  # <- default value
        
    )
    # Send the computed action `a` to the env.
    print(a)
    obs, reward, done, truncated, _ = env.step(a)
    episode_reward += reward
    # Is the episode `done`? -> Reset.
    if done:
        print(f"Episode {num_episodes} done: Total reward = {episode_reward}")
        obs, info = env.reset()
        num_episodes += 1
        episode_reward = 0.0
        state = init_state

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