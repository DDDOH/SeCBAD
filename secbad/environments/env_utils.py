import gym
from .mujoco.ant_dir import AntDir
from .hvac import HVAC
import numpy as np

from ..utils.helpers import RunningMeanStd


# [ ] A more general context class? Incoporating the context like OAT?
#  (at least we can test how varibad or sacbad works on context changing slowly like OAT?)


# class PiecewiseContext():


class SegmentContextDist():
    def __init__(self, sample_func, message=''):
        self.sample_func = sample_func
        self.message = ''

    def sample(self, length):
        return self.sample_func(length)

    def __repr__(self) -> str:
        return self.message


class NonstationaryContext():
    def __init__(self, traj_length, get_context_length, get_context_distribution):
        # traj_length: the length of the trajectory
        # get_context_length: a function that returns a random length, within these steps, context keep unchanged

        self.traj_length = traj_length
        self.get_context_length = get_context_length
        self.get_context_distribution = get_context_distribution

    def sample_traj_context(self, n_traj=1):
        traj_context_ls = []
        for i in range(n_traj):
            filled_length = 0
            traj_context = []

            context_length = self.get_context_length()
            context_distribution = self.get_context_distribution()

            while True:
                segment_context = context_distribution(context_length)
                traj_context.append(segment_context)
                filled_length += context_length

                if filled_length > self.traj_length:
                    break

                context_length = self.get_context_length()
                context_distribution = self.get_context_distribution()
            traj_context_ls.append(np.concatenate(
                traj_context)[:self.traj_length])

        return traj_context_ls


class VectorEnv():
    def __init__(self, env_name, n_env, traj_len, norm_rew):
        self.env_name = env_name
        self.n_env = n_env
        self.traj_len = traj_len
        if norm_rew:
            self.ret_rms = RunningMeanStd()
        self._make_vector_env()

    def get_env_paras(self):
        return self.env_paras

    def _make_vector_env(self):

        env_class = globals()[self.env_name]
        self.vec_env = gym.vector.AsyncVectorEnv([
            lambda: env_class(traj_len=self.traj_len, id=i) for i in range(self.n_env)
        ])

        self.env = env_class(traj_len=self.traj_len)

        self.env_paras = {'dim_action': self.env.action_space.shape[0],
                          'action_space': self.env.action_space,
                          'dim_context': self.env.dim_context if hasattr(self.env, 'dim_context') else -1,
                          'dim_state': self.env.observation_space.shape[0],
                          'max_trajectory_len': self.env.traj_len}

    def reset(self, traj_context_ls, seed=None):
        assert len(traj_context_ls) == self.n_env
        options_dict = {'traj_context': {}}
        for i in range(self.n_env):
            options_dict['traj_context'][i] = traj_context_ls[i]
        init_obs = self.vec_env.reset(seed=seed, options=options_dict)
        return init_obs
