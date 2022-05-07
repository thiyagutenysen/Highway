MODEL_NAME = "dqn____25.24mean_return__04-05-22_23-46-21"
MODEL_PATH = "C:\\Users\\thiya\\Desktop\\DDP\\openai gym\\highway2\\models"
import os

video_folder = os.path.join("result", MODEL_NAME)
import glob

videos = glob.glob(video_folder + "/*.mp4")
print(videos)
