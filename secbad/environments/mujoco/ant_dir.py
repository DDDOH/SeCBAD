'''
inherit from https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant_v4.py
rewrite step and reset_model to support nonstationary tasks
'''


from distutils.log import info
from tkinter.tix import Y_REGION
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.mujoco.ant_v4 import AntEnv
# from ..env_utils import SegmentContextDist, NonstationaryContext
import gym
import skvideo.io
# what we expect the env to do
# while training:
#   [ ] multi threading
#   [x] accept trajectory task when reset env
# while evaluating:
#   [x] single threading
#   [x] accept trajectory task when reset env (randomly chosen or read from files)
#   [x] rendering into video, camera following the agent
#   [x] set random seed
#   [ ] random seed test method
#   [ ] calculate traj, then render video as window


class AntDir(AntEnv):
    def __init__(self, traj_len, id=None):
        # https://www.runoob.com/w3cnote/python-extends-init.html
        self.init_parent = False  # whether AntEnv has been initialized
        """
        init AntEnv will actually call the gym.envs.mujoco.mujoco_env.py init method the step method is called as below
        '''
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)
        '''
        we only need assert not done and return observation of appropriate shape

        """
        AntEnv.__init__(self)
        self.init_parent = True  # whether AntEnv has been initialized
        self.id = id  # when multiple environments are created in VectorEnvs, the id is used to distinguish them
        self.traj_len = traj_len
        self.dim_context = 1

    # rewrite step function to recompute the reward function

    def step(self, action):
        if self.init_parent:
            xy_position_before = self.get_body_com("torso")[:2].copy()
            self.do_simulation(action, self.frame_skip)
            xy_position_after = self.get_body_com("torso")[:2].copy()

            agent_velocity = (xy_position_after - xy_position_before) / self.dt

            context = self.traj_context[self.traj_step]

            goal_direction = np.array([np.cos(context), np.sin(context)])
            forward_velocity = np.dot(agent_velocity, goal_direction)
            vertical_velocity = np.sqrt(
                agent_velocity[0]**2 + agent_velocity[1]**2 - forward_velocity**2)
            forward_reward = forward_velocity

            # forward_reward = x_velocity
            healthy_reward = self.healthy_reward

            rewards = forward_reward - 0.5 * vertical_velocity + healthy_reward

            costs = ctrl_cost = self.control_cost(action)

            # done = self.done
            observation = self._get_obs()
            info = {
                "reward_forward": forward_reward,
                "reward_ctrl": -ctrl_cost,
                "reward_survive": healthy_reward,
                "x_position": xy_position_after[0],
                "y_position": xy_position_after[1],
                # "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
                # "x_velocity": x_velocity,
                # "y_velocity": y_velocity,
            }
            if self._use_contact_forces:
                contact_cost = self.contact_cost
                costs += contact_cost
                info["reward_ctrl"] = -contact_cost

            reward = rewards - costs

            self.traj_step += 1

            if self.traj_step == self.traj_len:
                done = True
            else:
                done = False

            # 6-11 the new render API is not supported yet
            # self.renderer.render_step()
            return observation, reward, done, info
        else:
            observation = self._get_obs()
            done = False

            return observation, 0, done, {}

    # rewrite reset_model function to set the traj_context

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale *
            self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def reset(self, options=None, seed=None):
        # print(options)
        observation = super(AntEnv, self).reset(seed=seed)
        if options != None:
            traj_context = options['traj_context'][self.id]
            assert self.traj_len == len(
                traj_context), "traj_context length must be equal to max_episode_steps"
        else:
            traj_context = self.get_traj_context(self.traj_len)
        self.traj_context = traj_context
        self.traj_step = 0
        return observation

    def vis_traj(self, prev_obs, actions, rewards, info_rec, fig_dir_name):

        x_position = [_['x_position'] for _ in info_rec]
        y_position = [_['y_position'] for _ in info_rec]

        plt.figure(figsize=(3*5, 5))

        # position_x & pos_y
        plt.subplot(131)
        plt.scatter(x_position, y_position, c=np.arange(self.traj_len), s=1)

        plt.subplot(132)
        # task angle & step
        # true angle & step

        plt.subplot(133)
        # velocity along task angle

        plt.savefig(fig_dir_name)
        plt.close('all')
        # unchanged_length & step

    @ staticmethod
    def get_traj_context(traj_len):
        """Use goal direction as context

        Args:
            traj_len (_type_): _description_

        Returns:
            _type_: _description_
        """
        non_stationary = False

        if non_stationary:
            filled_length = 0
            traj_context = []

            context_length = int(max(np.random.normal(80, 20), 20))
            goal_direction_base = np.random.uniform(0, np.pi * 2)

            while True:
                segment_context = np.random.normal(
                    goal_direction_base, 0.2, context_length)
                traj_context.append(segment_context)
                filled_length += context_length
                if filled_length > traj_len:
                    break

                context_length = int(max(np.random.normal(80, 20), 20))
                goal_direction_base = np.random.uniform(0, np.pi * 2)

            return np.concatenate(traj_context)[:traj_len]
        else:
            goal_direction = np.random.uniform(0, np.pi * 2)
            return np.ones(traj_len) * goal_direction

    @staticmethod
    def get_context_distribution():
        # return a function that return a context
        goal_direction_base = np.random.uniform(0, np.pi * 2)

        def get_context(length):
            get_context.message = "np.random.normal({}, 0.1, length)".format(
                goal_direction_base)
            return np.random.normal(goal_direction_base, 0.1, length)

        return get_context


def diff_ratio(array_1, array_2):
    assert array_1.shape == array_2.shape
    return np.sum(array_1 - array_2) / array_1.size


def array_to_video(rgb_array, video_path_name):
    skvideo.io.vwrite(video_path_name + '.mp4', rgb_array)


# if __name__ == '__main__':

#     traj_len = 500
#     env = AntDir(traj_len=traj_len)

#     # vec_env = gym.vector.AsyncVectorEnv([lambda: gym.make(
#     #     id='non_envs/AntDir-v4', traj_len=100) for _ in range(3)])

#     vec_env = gym.vector.AsyncVectorEnv(
#         [lambda: AntDir(traj_len=traj_len) for _ in range(5)])

#     def get_context_length():
#         return int(max(np.random.normal(80, 20), 10))

#     non_context = NonstationaryContext(
#         traj_length=traj_len, get_context_length=get_context_length, get_context_distribution=AntDir.get_context_distribution)
#     traj_context = non_context.sample_traj_context()

#     env.reset(traj_context=traj_context, seed=10)
#     env.action_space.seed(seed=10)
#     render_rec_1 = []
#     obs_rec_1 = []
#     reward_rec_1 = []
#     for _ in range(traj_len):
#         observation, reward, done, info = env.step(env.action_space.sample())
#         render_rec_1.append(env.render(mode='rgb_array'))
#         obs_rec_1.append(observation)
#         reward_rec_1.append(reward)

#     env.reset(traj_context=traj_context, seed=10)
#     env.action_space.seed(seed=10)
#     render_rec_2 = []
#     obs_rec_2 = []
#     reward_rec_2 = []
#     for _ in range(traj_len):
#         observation, reward, done, info = env.step(env.action_space.sample())
#         render_rec_2.append(env.render(mode='rgb_array'))
#         obs_rec_2.append(observation)
#         reward_rec_2.append(reward)

#     render_rec_1 = np.array(render_rec_1)
#     render_rec_2 = np.array(render_rec_2)
#     obs_rec_1 = np.array(obs_rec_1)
#     obs_rec_2 = np.array(obs_rec_2)
#     reward_rec_1 = np.array(reward_rec_1)
#     reward_rec_2 = np.array(reward_rec_2)
#     print(diff_ratio(render_rec_1, render_rec_2))
#     print(diff_ratio(obs_rec_1, obs_rec_2))
#     print(diff_ratio(reward_rec_1, reward_rec_2))

#     array_to_video(render_rec_1, 'render_rec_1')
#     array_to_video(render_rec_2, 'render_rec_2')
