import gym
from .mujoco.ant_dir import AntDir
from .hvac import HVAC
import numpy as np
import os

from ..utils.helpers import RunningMeanStd, squash_action

# [ ] A more general context class? Incoporating the context like OAT?
#  (at least we can test how varibad or secbad works on context changing slowly like OAT?)


# class PiecewiseContext():


# class SegmentContextDist():
#     def __init__(self, sample_func, message=''):
#         self.sample_func = sample_func
#         self.message = ''

#     def sample(self, length):
#         return self.sample_func(length)

#     def __repr__(self) -> str:
#         return self.message


# class NonstationaryContext():
#     def __init__(self, traj_length, get_context_length, get_context_distribution):
#         # traj_length: the length of the trajectory
#         # get_context_length: a function that returns a random length, within these steps, context keep unchanged

#         self.traj_length = traj_length
#         self.get_context_length = get_context_length
#         self.get_context_distribution = get_context_distribution

#     def sample_traj_context(self, n_traj=1):
#         traj_context_ls = []
#         for i in range(n_traj):
#             filled_length = 0
#             traj_context = []

#             context_length = self.get_context_length()
#             context_distribution = self.get_context_distribution()

#             while True:
#                 segment_context = context_distribution(context_length)
#                 traj_context.append(segment_context)
#                 filled_length += context_length

#                 if filled_length > self.traj_length:
#                     break

#                 context_length = self.get_context_length()
#                 context_distribution = self.get_context_distribution()
#             traj_context_ls.append(np.concatenate(
#                 traj_context)[:self.traj_length])

#         return traj_context_ls


class NonstationaryContext():
    def __init__(self, get_traj_context, vis_iter_ls, traj_len):
        self.get_traj_context = get_traj_context
        self.vis_iter_ls = vis_iter_ls
        self.traj_len = traj_len

    def sample_context(self, n_traj):
        # sample context for training
        if n_traj == 1:
            return self.get_traj_context(self.traj_len)
        else:
            raise NotImplemented

    def load_context(self, file_dir_name, iter_idx, force_reset=False):
        # save (or load) context for visualization steps
        # get a empty dict, the key is vis_iter
        """_summary_

        Returns:
            force_reset: whether ignore the existing context file and get a new one
        """
        if (not os.path.exists(file_dir_name)) or force_reset:
            context_dict = {}
            for vis_iter in self.vis_iter_ls:
                context_dict[vis_iter] = self.sample_context(1)
            np.save(file_dir_name, context_dict)
        else:
            context_dict = np.load(file_dir_name, allow_pickle=True).item()
            assert set(context_dict.keys()) == set(self.vis_iter_ls)
        return context_dict[iter_idx]


class VectorEnv():
    def __init__(self, env_name, n_env, traj_len, norm_rew):
        self.env_name = env_name
        self.n_env = n_env
        self.traj_len = traj_len
        self.env_class = globals()[self.env_name]
        if norm_rew:
            self.ret_rms = RunningMeanStd()
        self._make_vector_env()
        self.test_env = self.env_class(traj_len=self.traj_len, id='test')
        self.get_traj_context = self.test_env.get_traj_context

    def get_env_paras(self):
        return self.env_paras

    def _make_vector_env(self):
        print('Making vector env')
        # self.vec_env = gym.vector.AsyncVectorEnv([
        #     lambda: self.env_class(traj_len=self.traj_len, id=i) for i in range(self.n_env)
        # ])

        self.vec_env = gym.vector.SyncVectorEnv([
            lambda: self.env_class(traj_len=self.traj_len, id=i) for i in range(self.n_env)
        ])

        # only to get env paras
        tmp_env = self.env_class(traj_len=self.traj_len)

        self.env_paras = {'dim_action': tmp_env.action_space.shape[0],
                          'action_space': tmp_env.action_space,
                          'dim_context': tmp_env.dim_context if hasattr(tmp_env, 'dim_context') else -1,
                          'dim_state': tmp_env.observation_space.shape[0],
                          'max_trajectory_len': tmp_env.traj_len}

    def reset(self, traj_context_ls, seed=None):
        assert len(traj_context_ls) == self.n_env
        options_dict = {'traj_context': {}}
        for i in range(self.n_env):
            options_dict['traj_context'][i] = traj_context_ls[i]
        init_obs = self.vec_env.reset(seed=seed, options=options_dict)
        return init_obs

    def reset_test(self, traj_context, seed=None):
        # traj_context = options['traj_context'][self.id]
        init_obs = self.test_env.reset(
            seed=seed, options={'traj_context': {self.test_env.id: traj_context}})
        return init_obs

    def test_env_step(self, action, args):
        # copied from utils.helpers.env_step

        act = squash_action(action, args)
        next_obs, reward, done, infos = self.test_env.step(act)

        # if isinstance(next_obs, list):
        #     next_obs = [o.to(device) for o in next_obs]
        # else:
        #     next_obs = next_obs.to(device)
        # if isinstance(reward, list):
        #     reward = [r.to(device) for r in reward]
        # else:
        #     reward = reward.to(device)

        # belief = torch.from_numpy(env.get_belief()).float().to(
        #     device) if args.pass_belief_to_policy else None
        # task = torch.from_numpy(env.get_task()).float().to(device) if (
        #     args.pass_task_to_policy or args.decode_task) else None

        belief = None
        task = None
        # return [next_obs, belief, task], reward, done, infos
        normalized_reward = None
        return next_obs, [reward, normalized_reward], done, infos

    def train_env_step(self, action, args):
        act = squash_action(action, args).numpy()
        next_obs, reward, done, infos = self.vec_env.step(act)

        # print(self.vec_env.envs[0].step_idx, self.vec_env.envs[1].step_idx)

        # if isinstance(next_obs, list):
        #     next_obs = [o.to(device) for o in next_obs]
        # else:
        #     next_obs = next_obs.to(device)
        # if isinstance(reward, list):
        #     reward = [r.to(device) for r in reward]
        # else:
        #     reward = reward.to(device)

        # belief = torch.from_numpy(env.get_belief()).float().to(
        #     device) if args.pass_belief_to_policy else None
        # task = torch.from_numpy(env.get_task()).float().to(device) if (
        #     args.pass_task_to_policy or args.decode_task) else None

        belief = None
        task = None
        # return [next_obs, belief, task], reward, done, infos
        normalized_reward = None
        return next_obs, [reward, normalized_reward], done, infos

    def vis_traj(self, prev_obs, actions, rewards, info_rec, file_dir_name):
        if hasattr(self.test_env, 'vis_traj'):
            self.test_env.vis_traj(prev_obs, actions, rewards, info_rec, file_dir_name)
        else:
            print('No vis_traj function')
