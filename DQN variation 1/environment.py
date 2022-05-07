import sys
import glob
import os
import numpy as np
import itertools
import time
import random
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import gym

# HYPER PARAMETERS

# Environment Code
class gym_env(py_environment.PyEnvironment):
    def __init__(self):
        # establish connection to the server
        self.env = gym.make("highway-v0")
        self.env.config["vehicles_density"] = 1.5
        self.sim = self.env
        self.actions = list(range(self.env.action_space.n))

        # PYENV
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=4, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(25,),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name="observation",
        )
        self._state = np.zeros(shape=(25,))
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def visually_simulate(self):
        self.env.render()

    def _reset(self):
        self._episode_ended = False
        state = self.env.reset()
        return ts.restart(np.array(state.flatten(), dtype=np.float32))

    def _step(self, action):
        # convert neural network max output neuron index to environment action
        # print(action)
        action = self.actions[action]

        # control the vehicle by giving inputs
        state, reward, done, extra = self.env.step(action)

        self._episode_ended = done
        if self._episode_ended:
            return (
                ts.termination(
                    np.array(state.flatten(), dtype=np.float32), reward=reward
                ),
            )
        else:
            return ts.transition(
                np.array(state.flatten(), dtype=np.float32),
                reward=reward,
                discount=1.0,
            )
        # return state.flatten(), reward, done, None
