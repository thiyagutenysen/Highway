import gym
import highway_env
import os
import random

# os.environ["SDL_VIDEODRIVER"] = "dummy"
# from gym.wrappers import RecordVideo
from matplotlib import pyplot as plt

env = gym.make("highway-v0")
env.config["observation"]["type"] = "OccupancyGrid"
env.config["observation"]["grid_step"] = [5, 5]
env.config["observation"]["features"] = ["presence"]
print("0000000000000000000000001111111111111", env.config["observation"]["grid_step"])
env.config["vehicles_density"] = 1
print(env.action_space)
env.reset()
action = 0
for _ in range(20):
    # action = env.action_type.actions_indexes["IDLE"]
    # action += 1
    # obs, reward, done, info = env.step(action % 5)
    obs, reward, done, info = env.step(env.action_space.sample())
    print(action, obs.shape, type(obs), done, reward)
    env.render()

# plt.imshow(env.render(mode="rgb_array"))
# plt.show()
