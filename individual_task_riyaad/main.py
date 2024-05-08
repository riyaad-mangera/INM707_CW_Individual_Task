import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from ray.tune.registry import register_env
import matplotlib.pyplot as plt


def preprocess_env(env):
    env = gym.wrappers.AtariPreprocessing(env, noop_max = 30, frame_skip = 4, screen_size = 84, 
                                                  terminal_on_life_loss = False, grayscale_obs = True, 
                                                  grayscale_newaxis = False, scale_obs = False)
    
    env = gym.wrappers.FrameStack(env, 4)

    return env

def create_env():

    return gym.make("BreakoutNoFrameskip-v4")

# env = preprocess_env(gym.make("BreakoutNoFrameskip-v4", render_mode="human"))

ray.init()
register_env("BreakoutNoFrameskip-v4", lambda config: create_env())

algo = ppo.PPO(env="BreakoutNoFrameskip-v4")
# agent = ppo.PPOTrainer(env=env)

# env.reset()
# n_actions = env.action_space.n
# state_dim = env.observation_space.shape

for _ in range(1):
    print(_)
    print(algo.train())
    #obs = env.step(env.action_space.sample())