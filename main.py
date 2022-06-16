import argparse
import warnings
import numpy as np
import torch
from secbad.environments.mujoco.ant_dir import AntDir

import os

import platform
if platform.system() == 'Linux':
    os.environ['MUJOCO_GL'] = "egl"

from secbad.config.mujoco import args_half_dir_non_varibad, args_half_dir_non_varibad_xt, args_half_dir_non_varibad_single, \
    args_half_dir_non_sacbad
from secbad.config.mujoco import args_half_goal_non_varibad, args_half_goal_non_varibad_xt, args_half_goal_non_varibad_single, \
    args_half_goal_non_sacbad
from secbad.config.mujoco import args_ant_goal_non_varibad, args_ant_goal_non_varibad_xt, args_ant_goal_non_varibad_single, \
    args_ant_goal_non_sacbad
from secbad.config.mujoco import args_ant_dir_non_varibad, args_ant_dir_non_varibad_xt, args_ant_dir_non_varibad_single, \
    args_ant_dir_non_sacbad
from secbad.config.mujoco import args_ant_vel_non_varibad, args_ant_vel_non_varibad_xt, args_ant_vel_non_varibad_single, \
    args_ant_vel_non_sacbad
from secbad.config.mujoco import args_half_vel_non_sacbad

from secbad.config import args_hvac_varibad


########################## archive ##########################
# get configs
from secbad.config.gridworld import \
    args_grid_belief_oracle, args_grid_rl2, args_grid_varibad, args_grid_nonstationary
# from config.pointrobot import \
#     args_pointrobot_multitask, args_pointrobot_varibad, args_pointrobot_rl2, args_pointrobot_humplik
########################## archive ##########################


# from environments.parallel_envs import make_vec_envs
# [ ] move learners into learners folder
from secbad.learner import Learner
from secbad.metalearner import MetaLearner
from secbad.oracle_truncate_learner import OracleTruncateLearner
from secbad.adaptive_learner import AdaptiveLearner


def main():
    DEBUG = True
    CUDA = torch.cuda.is_available()
    CUDA_COUNT = torch.cuda.device_count()

    if CUDA and not DEBUG:
        NUM_PROCESSES = 24
        LOG_INTERVAL = 100  # 这个并不决定 test evaluation 的频率，他只是用 online 搜集到的数据算一个 training 时候的 reward
    else:
        NUM_PROCESSES = 2
        LOG_INTERVAL = 50
        NUM_FRAMES = 100000

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--env-type', default='args_hvac_varibad')

    args, rest_args = parser.parse_known_args()
    env = args.env_type

    args = globals()[env].get_args(rest_args)

    # overwrite settings in config folder
    if DEBUG:
        args.exp_label = 'debug_' + args.exp_label
        args.num_frames = NUM_FRAMES
    args.num_processes = NUM_PROCESSES
    args.log_interval = LOG_INTERVAL

    if CUDA:
        if CUDA_COUNT == 1:  # for A100
            args.results_log_dir = '/home/yufeng/SaCBAD_tmp'
        else:  # for V100
            args.results_log_dir = '/home/yufeng/SaCBAD_tmp'
    else:
        args.results_log_dir = '/Users/shuffleofficial/Offline_Documents/SaCBAD/tmp_results'

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # if we're normalising the actions, we have to make sure that the env expects actions within [-1, 1]
    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None,
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    # clean up arguments
    if args.learner_type == 'ori' or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed

    # envs = gym.vector.AsyncVectorEnv([lambda: gym.make(
    #     id=args.env_name, traj_len=args.max_episode_steps) for _ in range(args.num_processes)])

    # envs = gym.vector.AsyncVectorEnv([
    #     lambda: AntDir(traj_len=500),
    #     lambda: AntDir(traj_len=500),
    #     lambda: AntDir(traj_len=500)
    # ])

    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        # # "select from ori, meta, orical_truncate, sacbad"
        # if args.learner_type == 'sacbad':
        #     learner = AdaptiveLearner(args)
        # elif args.learner_type == 'varibad':
        #     learner = MetaLearner(args)
        # elif args.learner_type == 'ori':
        #     learner = Learner(args)
        # elif args.learner_type == 'oracle_truncate':
        #     learner = OracleTruncateLearner(args)
        # else:
        #     raise Exception("Invalid Learner Type")

        # put all type learners together

        learner = AdaptiveLearner(args)

        # if args has attribute visualize_model
        if hasattr(args, 'visualize_model') and args.visualize_model:
            learner.visualize()
        else:
            learner.train()


if __name__ == '__main__':
    main()
