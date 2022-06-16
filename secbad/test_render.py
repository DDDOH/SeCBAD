# import skvideo.io
# import numpy as np

# render_rec = np.array(np.load('render_rec.npy', allow_pickle=True))


# # we create frames using numpy random method
# outputdata = np.random.random(size=(5, 720, 1280, 3)) * 255
# outputdata = outputdata.astype(np.uint8)

# # save array to output file
# skvideo.io.vwrite("outputvideo.mp4", render_rec)


import gym
import non_envs

if __name__ == '__main__':
    env = gym.vector.AsyncVectorEnv([lambda: gym.make(
        id='non_envs/AntDir-v4', traj_len=100) for _ in range(3)])


    env.reset()