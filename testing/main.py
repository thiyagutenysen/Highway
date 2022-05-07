# Hyper Parameters
TEST_EPISODES = 5
# model name
MODEL_NAME = "actor____29.96mean_return__03-05-22_18-42-22"
MODEL_PATH = "C:\\Users\\thiya\\Desktop\\DDP\\openai gym\\highway1\\models"

if __name__ == "__main__":

    import gym
    import highway_env
    import os
    from tqdm import tqdm
    from tensorflow import keras
    from environment import gym_env
    from tf_agents.environments import tf_py_environment
    import tensorflow as tf
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    from threading import Thread
    import time

    # from gym_recording.wrappers import Monitor

    # env = gym.make("highway-v0")
    env = gym_env()
    tf_env = tf_py_environment.TFPyEnvironment(env)

    video_folder = os.path.join("result", MODEL_NAME)
    # env = gym.wrappers.RecordVideo(
    #     env, video_folder=video_folder, episode_trigger=lambda e: True
    # )
    # env.unwrapped.set_record_video_wrapper(env)
    # env = Monitor(env, video_folder, video_callable=lambda episode: True, force=True)

    # create directory to save model specific test data
    if not os.path.isdir(os.path.join("result", MODEL_NAME)):
        os.mkdir(os.path.join("result", MODEL_NAME))

    # video_recorder = VideoRecorder(
    #     env.env, os.path.join(video_folder, "video.mp4"), enabled=True
    # )
    def parallel_run():
        while flag:
            time.sleep(1 / 30)
            env.env.unwrapped.render()
            video_recorder.capture_frame()

    actor_model = tf.compat.v2.saved_model.load(os.path.join(MODEL_PATH, MODEL_NAME))
    # env.env.metadata["render_fps"] = 120
    for episode in tqdm(range(TEST_EPISODES)):
        env.episode = episode
        time_step = tf_env.reset()
        # env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
        done = False
        # env.visually_simulate()
        flag = True
        video_recorder = VideoRecorder(
            env.env,
            os.path.join(video_folder, "video" + str(episode + 1) + ".mp4"),
            # metadata={"render_fps": 120},
            enabled=True,
        )
        training_thread = Thread(target=parallel_run, daemon=True)
        training_thread.start()
        while not time_step.is_last():
            # env.env.unwrapped.render()
            # video_recorder.capture_frame()
            action_step = actor_model.action(time_step)
            time_step = tf_env.step(action_step.action)
            env.visually_simulate()
        flag = False
        training_thread.join()
        video_recorder.close()
        video_recorder.enabled = False
    env.close()

    # fasten the videos
    import cv2
    import glob

    i = 1
    videos = glob.glob(video_folder + "/*.mp4")
    for video in videos:
        cap = cv2.VideoCapture(video)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            video_folder + "/output" + str(i) + ".avi",
            fourcc,
            300.0,
            (600, 150),
        )
        i += 1
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.flip(frame, 0)
                out.write(frame)

                # cv2.imshow("frame", frame)
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
