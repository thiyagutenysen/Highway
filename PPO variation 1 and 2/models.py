import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf_agents
from tf_agents.networks import sequential
import main
from environment import gym_env
from tf_agents.environments import tf_py_environment
import gym
import highway_env
from tf_agents.environments import suite_gym
from tf_agents.networks import actor_distribution_network, value_network

# We need to define 2 networks
# 1. Actor network - input: state, output: action, probability
# 2. critique network - input: state, output: Value

# Hyper Parameters
env = gym_env()
# tf_env = suite_gym.load("highway-v0")
tf_env = tf_py_environment.TFPyEnvironment(env)

# Maintain consistency
tf.random.set_seed(6)

# # essential code for setting memory growth on gpu to effeciently use gpu memory
# physical_devices = tf.config.list_physical_devices("GPU")
# print("Available GPUs =", physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

actor_fc_layers = (128, 128)
value_fc_layers = (128, 128)

Actor = actor_distribution_network.ActorDistributionNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    fc_layer_params=actor_fc_layers,
    activation_fn=tf.keras.activations.tanh,
)

Critique = value_network.ValueNetwork(
    tf_env.observation_spec(),
    fc_layer_params=value_fc_layers,
    activation_fn=tf.keras.activations.tanh,
)
