import gym
import time
import highway_env

env = gym.make("highway-v0")

# env = RecordVideo(
#     env, video_folder="run", episode_trigger=lambda e: True
# )  # record all episodes

# # Provide the video recorder to the wrapped environment
# # so it can send it intermediate simulation frames.
# env.unwrapped.set_record_video_wrapper(env)

env.reset()

for _ in range(1000):
    env.render()
    time.sleep(0.1)
    env.step(env.action_space.sample())

env.close()
