import tensorflow as tf
from models import Actor, Critique
from environment import gym_env
import os
import time
import datetime
import tf_agents
from tf_agents.environments import tf_py_environment
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.policies import policy_saver
from tensorflow.keras.optimizers.schedules import ExponentialDecay, PolynomialDecay
import main
import gym
import highway_env
from tf_agents.environments import suite_gym

# # essential code for setting memory growth on gpu to effeciently use gpu memory
# physical_devices = tf.config.list_physical_devices("GPU")
# print("Available GPUs =", physical_devices)
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Hyper Parameters
MODEL_DIR = main.MODEL_DIR
MODEL_NAME = main.MODEL_NAME

env = gym_env()
# tf_env = suite_gym.load("highway-v0")
tf_env = tf_py_environment.TFPyEnvironment(env)
actor = Actor
critique = Critique

log_folder = "log_dir"
tensorboard_writer = tf.summary.create_file_writer(
    log_folder
    + "/"
    + MODEL_NAME
    + "_"
    + datetime.datetime.now().strftime("%d-%m-%y_%H-%M-%S")
    + "/"
)

agent = tf_agents.agents.PPOClipAgent(
    time_step_spec=tf_env.time_step_spec(),
    action_spec=tf_env.action_spec(),
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),  # ExponentialDecay(initial_learning_rate=0.001,decay_rate=0.8,decay_steps=300)),
    actor_net=actor,
    value_net=critique,
    greedy_eval=False,
    importance_ratio_clipping=0.2,
    lambda_value=0.95,
    discount_factor=0.99,
    entropy_regularization=0.01,
    num_epochs=1,
    use_gae=True,
    use_td_lambda_return=False,
    normalize_rewards=False,
    normalize_observations=True,
    debug_summaries=False,  # train_step_counter = tf.Variable(0)
)

agent.initialize()
print(actor.summary())


def save_models(average_season_reward):
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        time.sleep(2)
    time_str = str(int(time.time()))
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    policy_dir = f"{MODEL_DIR}/actor__{average_season_reward:_>7.2f}mean_return__{datetime.datetime.now().strftime('%d-%m-%y_%H-%M-%S')}"
    tf_policy_saver.save(policy_dir)
