# HYPER PARAMETERS
REPLAY_BUFFER_MAX_LENGTH = 500
SEASONS = 300
EPISODES = 10
MODEL_DIR = "models"
MODEL_NAME = "mine"
EPOCHS = 5

if __name__ == "__main__":
    from agent import agent, tf_env, save_models, tensorboard_writer, env
    import tensorflow as tf
    import tf_agents
    from tf_agents.replay_buffers import tf_uniform_replay_buffer
    from tf_agents.trajectories import trajectory
    from tqdm import tqdm
    import matplotlib.pyplot as plt

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

    global_episode_number = 0
    episode_rewards = []
    episode_losses = []
    best_season_score = float("-inf")
    for season in tqdm(
        range(1, SEASONS + 1), ascii=False, unit="season", desc="SEASONS"
    ):
        replay_buffer.clear()
        current_season_reward = 0
        average_season_reward = 0
        current_season_length = 0
        for episode in range(1, EPISODES + 1):
            env.episode = global_episode_number
            # collect data
            current_episode_timestep = 0
            time_step = tf_env.reset()
            current_episode_reward = 0
            while not time_step.is_last():
                action_step = agent.collect_policy.action(time_step)
                next_time_step = tf_env.step(action_step.action)
                current_episode_reward += next_time_step.reward.numpy()[0]
                traj = trajectory.from_transition(
                    time_step, action_step, next_time_step
                )
                replay_buffer.add_batch(traj)
                time_step = next_time_step
                current_episode_timestep += 1
            # print(actor.summary())
            global_episode_number += 1
            # env.destroy_actors()
            current_season_reward += current_episode_reward
            episode_rewards.append(current_episode_reward)
            current_season_length += current_episode_timestep
            with tensorboard_writer.as_default():
                tf.summary.scalar(
                    "Episode Reward",
                    data=current_episode_reward,
                    step=global_episode_number,
                )
                tf.summary.scalar(
                    "Episode Length",
                    data=current_episode_timestep,
                    step=global_episode_number,
                )
                # tf.summary.scalar(
                #     "Termination State",
                #     data=env.termination_state,
                #     step=global_episode_number,
                # )
                # tf.summary.scalar(
                #     "Distance Travelled along X-axis",
                #     data=env.distance_travelled,
                #     step=global_episode_number,
                # )
            # path = env.path
            # x = [coord[0] for coord in env.path]
            # y = [coord[1] for coord in env.path]
            # plt.figure()
            # plt.plot(x, y)
            # plt.xlabel("x - axis")
            # plt.ylabel("y - axis")
            # plt.axis("equal")
            # plt.savefig(f"paths/episode_{global_episode_number}.png")
        average_season_reward = current_season_reward / EPISODES
        average_season_episode_length = current_season_length / EPISODES
        if best_season_score < current_season_reward:
            save_models(average_season_reward)
            best_season_score = current_season_reward
        with tensorboard_writer.as_default():
            tf.summary.scalar(
                "Season Reward",
                data=current_season_reward,
                step=season,
            )
            tf.summary.scalar(
                "Average Episode Reward per Season",
                data=average_season_reward,
                step=season,
            )
            tf.summary.scalar(
                "Average Episode Length per Season",
                data=average_season_episode_length,
                step=season,
            )

        experience = replay_buffer.gather_all()
        # for epoch in range(EPOCHS):
        #     dataset = replay_buffer.as_dataset(
        #         sample_batch_size=8,
        #         num_steps=2,
        #         num_parallel_calls=2,
        #     ).prefetch(3)
        #     iterator = iter(dataset)
        #     for _ in range(REPLAY_BUFFER_MAX_LENGTH // 8):
        #         # iterator = iter(dataset)
        #         experience, unused_info = next(iterator)
        #         agent.train(experience)
        agent.train(experience)
