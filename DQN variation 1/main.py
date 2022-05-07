# HYPER PARAMETERS
REPLAY_BUFFER_MAX_LENGTH = 5000
EPISODES = 3000
MODEL_DIR = "models"
MODEL_NAME = "mine"
EPOCHS = 5
AGGREGATE_STATS_EVERY = 10
ACTIONS = 5
EPSILON_DECAY = 0.9975
epsilon = 1.0

if __name__ == "__main__":
    from agent import agent, tf_env, save_models, tensorboard_writer, env
    import tensorflow as tf
    import tf_agents
    from tf_agents.replay_buffers import tf_uniform_replay_buffer
    from tf_agents.trajectories import trajectory
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import numpy as np

    # # essential code for setting memory growth on gpu to effeciently use gpu memory
    # physical_devices = tf.config.list_physical_devices("GPU")
    # print("Available GPUs =", physical_devices)
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        # batch_size=8,
        max_length=REPLAY_BUFFER_MAX_LENGTH,
    )

    dataset = replay_buffer.as_dataset(
        sample_batch_size=128,
        num_steps=2,
        num_parallel_calls=2,
    ).prefetch(3)

    episode_rewards = []
    episode_losses = []
    best_season_score = float("-inf")
    global_time_step = 0
    agent.train_step_counter.assign(0)
    for episode in range(1, EPISODES + 1):
        # collect data
        step = agent.train_step_counter.numpy()
        current_episode_timestep = 0
        time_step = tf_env.reset()
        current_episode_reward = 0
        epsilon = np.max([0.001, epsilon * EPSILON_DECAY])
        while not time_step.is_last():
            # epsilon = np.maximum(1 - (episode / EPISODES), 0)
            # epsilon = np.max([0.001, epsilon * EPSILON_DECAY])
            if np.random.random() < epsilon:
                action_no = np.random.randint(0, ACTIONS)
                action = tf.constant(
                    [action_no], shape=(1,), dtype=np.int32, name="action"
                )
                action_step = tf_agents.trajectories.policy_step.PolicyStep(action)
            else:
                action_step = agent.collect_policy.action(time_step)
            next_time_step = tf_env.step(action_step.action)
            current_episode_reward += next_time_step.reward.numpy()[0]
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            replay_buffer.add_batch(traj)
            time_step = next_time_step
            current_episode_timestep += 1
            global_time_step += 1

            # train
            if global_time_step % 20 == 0 and global_time_step > 500:
                iterator = iter(dataset)
                experience, _ = next(iterator)
                agent.train(experience)
        # train
        iterator = iter(dataset)
        experience, _ = next(iterator)
        episode_losses.append(agent.train(experience).loss)
        episode_rewards.append(current_episode_reward)
        with tensorboard_writer.as_default():
            tf.summary.scalar(
                "Episode Reward",
                data=current_episode_reward,
                step=episode,
            )
            tf.summary.scalar(
                "Episode Length",
                data=current_episode_timestep,
                step=episode,
            )
            tf.summary.scalar(
                "Episode Loss",
                data=episode_losses[-1],
                step=episode,
            )
            tf.summary.scalar(
                "Epsilon",
                data=epsilon,
                step=episode,
            )
        # average_season_reward = current_season_reward / EPISODES
        # average_season_episode_length = current_season_length / EPISODES
        if episode % AGGREGATE_STATS_EVERY == 0:
            last_n_episodes = episode_rewards[-AGGREGATE_STATS_EVERY:]
            average_reward = sum(last_n_episodes) / AGGREGATE_STATS_EVERY
            with tensorboard_writer.as_default():
                tf.summary.scalar(
                    "Average Episode Reward",
                    data=average_reward,
                    step=episode,
                )
                tf.summary.scalar(
                    "Minimum Episode Reward",
                    data=min(last_n_episodes),
                    step=episode,
                )
                tf.summary.scalar(
                    "Maximum Episode Reward",
                    data=max(last_n_episodes),
                    step=episode,
                )
            if best_season_score < average_reward:
                save_models(average_reward)
                best_season_score = average_reward
