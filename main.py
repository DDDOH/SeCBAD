"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""

# Done return 曲线，除了Max&min，加一个更细致的
# DONE 直接画10个latent，不搞那个mean & variance了

import argparse
import warnings
# from xml.etree.ElementTree import TreeBuilder

import numpy as np
import torch


from config.mujoco.supplement import args_half_dir_non_varibad, args_half_dir_non_varibad_xt, args_half_dir_non_varibad_single, args_half_dir_non_sacbad
from config.mujoco.supplement import args_half_goal_non_varibad, args_half_goal_non_varibad_xt, args_half_goal_non_varibad_single, args_half_goal_non_sacbad
from config.mujoco.supplement import args_ant_goal_non_varibad, args_ant_goal_non_varibad_xt, args_ant_goal_non_varibad_single, args_ant_goal_non_sacbad
from config.mujoco.supplement import args_ant_dir_non_varibad, args_ant_dir_non_varibad_xt, args_ant_dir_non_varibad_single, args_ant_dir_non_sacbad
from config.mujoco.supplement import args_ant_velocity_non_varibad, args_ant_velocity_non_varibad_xt, args_ant_velocity_non_varibad_single, args_ant_velocity_non_sacbad


# set target velocity on cheetah env, using three algorithms
from config.mujoco.archive import args_cheetah_vel_varibad, args_cheetah_vel_oracle_truncate, args_cheetah_vel_nonstationary

# wind sppeed changing, maximize forward velocity on cheetah env, using varibad and sacbad
from config.mujoco.archive import args_cheetah_wind_nonstationary, args_cheetah_wind_varibad

# wind speed changing, make the agent keep in place, using varibad and sacbad
from config.mujoco.archive import args_cheetah_wind_stay_nonstationary


# get configs
from config.gridworld import \
    args_grid_belief_oracle, args_grid_rl2, args_grid_varibad, args_grid_nonstationary
from config.pointrobot import \
    args_pointrobot_multitask, args_pointrobot_varibad, args_pointrobot_rl2, args_pointrobot_humplik


from config.mujoco.archive import \
    args_cheetah_dir_multitask, args_cheetah_dir_expert, args_cheetah_dir_rl2, args_cheetah_dir_varibad, \
    args_cheetah_vel_multitask, args_cheetah_vel_expert, args_cheetah_vel_rl2, \
    args_cheetah_vel_avg, args_ant_dir_multitask, args_ant_dir_expert, args_ant_dir_rl2, args_ant_dir_varibad, args_ant_dir_nonstationary, \
    args_ant_wind_varibad, args_ant_wind_nonstationary, \
    args_ant_goal_multitask, args_ant_goal_expert, args_ant_goal_rl2, args_ant_goal_varibad, \
    args_ant_goal_humplik, \
    args_walker_multitask, args_walker_expert, args_walker_avg, args_walker_rl2, args_walker_varibad, \
    args_humanoid_dir_varibad, args_humanoid_dir_rl2, args_humanoid_dir_multitask, args_humanoid_dir_expert
from environments.parallel_envs import make_vec_envs
from learner import Learner
from metalearner import MetaLearner
from oracle_truncate_learner import OracleTruncateLearner
from adaptive_learner import AdaptiveLearner


def main():

    DEBUG = False
    CUDA = torch.cuda.is_available()
    CUDA_COUNT = torch.cuda.device_count()

    if CUDA and not DEBUG:
        NUM_PROCESSES = 24
        LOG_INTERVAL = 100  # 这个并不决定 test evaluation 的频率，他只是用 online 搜集到的数据算一个 training 时候的 reward
    else:
        NUM_PROCESSES = 2
        LOG_INTERVAL = 50

    parser = argparse.ArgumentParser()
    # parser.add_argument('--env-type', default='gridworld_varibad')
    # parser.add_argument('--env-type', default='gridworld_nonstationary')

    # parser.add_argument('--env-type', default='cheetah_vel_varibad') # [x]
    # parser.add_argument('--env-type', default='cheetah_vel_nonstationary') # [x]
    # parser.add_argument('--env-type', default='cheetah_vel_oracle_truncate') # [x]
    # parser.add_argument('--env-type', default='cheetah_wind_nonstationary') # [x]
    # parser.add_argument('--env-type', default='cheetah_wind_varibad') # [x]

    # parser.add_argument('--env-type', default='ant_wind_varibad') # [x]
    # parser.add_argument('--env-type', default='ant_wind_nonstationary') # [x]
    # parser.add_argument('--env-type', default='ant_dir_varibad') # [x]
    # parser.add_argument('--env-type', default='ant_dir_nonstationary') # [x]

    parser.add_argument('--env-type', default='gridworld_varibad')  # [x]

    args, rest_args = parser.parse_known_args()
    env = args.env_type

    if env == 'cheetah_vel_varibad':
        args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == 'cheetah_vel_nonstationary':
        args = args_cheetah_vel_nonstationary.get_args(rest_args)
    elif env == 'cheetah_vel_oracle_truncate':
        args = args_cheetah_vel_oracle_truncate.get_args(rest_args)

    elif env == 'cheetah_wind_nonstationary':
        args = args_cheetah_wind_nonstationary.get_args(rest_args)
    elif env == 'cheetah_wind_varibad':
        args = args_cheetah_wind_varibad.get_args(rest_args)
    elif env == 'cheetah_wind_varibad':
        args = args_cheetah_wind_stay_nonstationary.get_args(rest_args)

    # --- GridWorld ---

    elif env == 'gridworld_belief_oracle':
        args = args_grid_belief_oracle.get_args(rest_args)
    elif env == 'gridworld_varibad':
        args = args_grid_varibad.get_args(rest_args)
    elif env == 'gridworld_rl2':
        args = args_grid_rl2.get_args(rest_args)
    elif env == 'gridworld_nonstationary':
        args = args_grid_nonstationary.get_args(rest_args)

    # --- PointRobot 2D Navigation ---

    elif env == 'pointrobot_multitask':
        args = args_pointrobot_multitask.get_args(rest_args)
    elif env == 'pointrobot_varibad':
        args = args_pointrobot_varibad.get_args(rest_args)
    elif env == 'pointrobot_rl2':
        args = args_pointrobot_rl2.get_args(rest_args)
    elif env == 'pointrobot_humplik':
        args = args_pointrobot_humplik.get_args(rest_args)

    # --- MUJOCO ---

    # - CheetahDir -
    elif env == 'cheetah_dir_multitask':
        args = args_cheetah_dir_multitask.get_args(rest_args)
    elif env == 'cheetah_dir_expert':
        args = args_cheetah_dir_expert.get_args(rest_args)
    elif env == 'cheetah_dir_varibad':
        args = args_cheetah_dir_varibad.get_args(rest_args)
    elif env == 'cheetah_dir_rl2':
        args = args_cheetah_dir_rl2.get_args(rest_args)
    #
    # - CheetahVel -
    elif env == 'cheetah_vel_multitask':
        args = args_cheetah_vel_multitask.get_args(rest_args)
    elif env == 'cheetah_vel_expert':
        args = args_cheetah_vel_expert.get_args(rest_args)
    elif env == 'cheetah_vel_avg':
        args = args_cheetah_vel_avg.get_args(rest_args)
    elif env == 'cheetah_vel_rl2':
        args = args_cheetah_vel_rl2.get_args(rest_args)
    #
    # - AntDir -
    elif env == 'ant_dir_multitask':
        args = args_ant_dir_multitask.get_args(rest_args)
    elif env == 'ant_dir_expert':
        args = args_ant_dir_expert.get_args(rest_args)
    elif env == 'ant_dir_varibad':
        args = args_ant_dir_varibad.get_args(rest_args)
    elif env == 'ant_dir_rl2':
        args = args_ant_dir_rl2.get_args(rest_args)
    elif env == 'ant_dir_nonstationary':
        args = args_ant_dir_nonstationary.get_args(rest_args)

    # - AntWind -
    elif env == 'ant_wind_varibad':
        args = args_ant_wind_varibad.get_args(rest_args)
    elif env == 'ant_wind_nonstationary':
        args = args_ant_wind_nonstationary.get_args(rest_args)

    #
    # - AntGoal -
    elif env == 'ant_goal_multitask':
        args = args_ant_goal_multitask.get_args(rest_args)
    elif env == 'ant_goal_expert':
        args = args_ant_goal_expert.get_args(rest_args)
    elif env == 'ant_goal_varibad':
        args = args_ant_goal_varibad.get_args(rest_args)
    elif env == 'ant_goal_humplik':
        args = args_ant_goal_humplik.get_args(rest_args)
    elif env == 'ant_goal_rl2':
        args = args_ant_goal_rl2.get_args(rest_args)
    #
    # - Walker -
    elif env == 'walker_multitask':
        args = args_walker_multitask.get_args(rest_args)
    elif env == 'walker_expert':
        args = args_walker_expert.get_args(rest_args)
    elif env == 'walker_avg':
        args = args_walker_avg.get_args(rest_args)
    elif env == 'walker_varibad':
        args = args_walker_varibad.get_args(rest_args)
    elif env == 'walker_rl2':
        args = args_walker_rl2.get_args(rest_args)
    #
    # - HumanoidDir -
    elif env == 'humanoid_dir_multitask':
        args = args_humanoid_dir_multitask.get_args(rest_args)
    elif env == 'humanoid_dir_expert':
        args = args_humanoid_dir_expert.get_args(rest_args)
    elif env == 'humanoid_dir_varibad':
        args = args_humanoid_dir_varibad.get_args(rest_args)
    elif env == 'humanoid_dir_rl2':
        args = args_humanoid_dir_rl2.get_args(rest_args)
    else:
        raise Exception("Invalid Environment")

    # overwrite settings in config folder
    if DEBUG:
        args.exp_label = 'debug_' + args.exp_label
    args.num_processes = NUM_PROCESSES
    args.log_interval = LOG_INTERVAL

    if CUDA:
        if CUDA_COUNT == 1:  # for A100
            args.results_log_dir = '/home/v-yuzheng/tmp_results'
        else:  # for V100
            args.results_log_dir = '/home/yufeng/tmp_results'
    else:
        args.results_log_dir = '/Users/hector/Offline Documents/Latent_Adaptive_RL/tmp_results'

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
    if args.disable_metalearner or args.disable_decoder:
        # if args.learner_type == 'ori' or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        # "select from ori, meta, orical_truncate, adaptive")
        # if args.learner_type == 'adaptive':
        #     learner = AdaptiveLearner(args)
        # elif args.learner_type == 'varibad':
        #     learner = MetaLearner(args)
        # elif args.learner_type == 'ori':
        #     learner = Learner(args)
        # elif args.learner_type == 'oracle_truncate':
        #     learner = OracleTruncateLearner(args)
        # else:
        #     raise Exception("Invalid Learner Type")
        learner = MetaLearner(args)

        # if args.enable_adaptivelearner:
        #     learner = AdaptiveLearner(args)
        #     print('Use AdaptiveLearner')
        # elif args.disable_metalearner:
        #     # If `disable_metalearner` is true, the file `learner.py` will be used instead of `metalearner.py`.
        #     # This is a stripped down version without encoder, decoder, stochastic latent variables, etc.
        #     learner = Learner(args)
        #     print('Use original Learner')
        # else:
        #     learner = MetaLearner(args)
        #     print('Use MetaLearner')
        learner.train()


if __name__ == '__main__':
    main()


# DONE update gird env to non-stationary case (the reward location may change)
# TODO call the adaptive learner
