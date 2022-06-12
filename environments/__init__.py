# from gym.envs.registration import register

# register(
#     id='non_envs/AntDir-v4',
#     entry_point='non_envs.mujoco:AntDir',
#     # max_episode_steps=400,
# )

# Mujoco
# ----------------------------------------


# supplement

# register(
#     'HalfDirNon-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.supplement.half_dir_non:HalfDirNon',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'HalfGoalNon-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.supplement.half_goal_non:HalfGoalNon',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'AntGoalNon-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.supplement.ant_goal_non:AntGoalNon',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )


# register(
#     'AntDirNon-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.supplement.ant_dir_non:AntDirNon',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'AntVelNon-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.supplement.ant_vel_non:AntVelNon',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'HalfVelNon-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.supplement.half_vel_non:HalfVelNon',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )


# # archive

# # - randomised reward functions

# register(
#     'AntDir-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDirEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'AntDir2D-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_dir:AntDir2DEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200,
# )

# register(
#     'AntDir2DNonstationary-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_dir_nonstationary:AntDir2DNonstationary',
#             'max_episode_steps': 200},
#     max_episode_steps=200,
# )

# register(
#     'AntWindNonstationary-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_wind_nonstationary:AntWindNonstationary',
#             'max_episode_steps': 200},
#     max_episode_steps=200,
# )

# register(
#     'AntGoal-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.ant_goal:AntGoalEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'HalfCheetahDir-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.half_cheetah_dir:HalfCheetahDirEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'HalfCheetahVel-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel:HalfCheetahVelEnv',
#             'max_episode_steps': 40},
#     max_episode_steps=40
# )

# register(
#     'HalfCheetahVelNonstationary-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.half_cheetah_vel_nonstationary:HalfCheetahVelEnvNonstationary',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# # register(
# #     id="HalfCheetahWind-v0",
# #     entry_point="environments.mujoco.half_cheetah_wind_nonstationary:HalfCheetahWindEnv",
# #     max_episode_steps=1000,
# #     reward_threshold=4800.0, # reward_threshold (float) – Gym environment argument, the reward threshold before the task is considered solved
# # )

# register(
#     'HalfCheetahWindNonstationary-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.half_cheetah_wind_nonstationary:HalfCheetahWindNonstationary',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# register(
#     'HumanoidDir-v0',
#     entry_point='environments.wrappers:mujoco_wrapper',
#     kwargs={'entry_point': 'environments.mujoco.humanoid_dir:HumanoidDirEnv',
#             'max_episode_steps': 200},
#     max_episode_steps=200
# )

# # - randomised dynamics

# register(
#     id='Walker2DRandParams-v0',
#     entry_point='environments.mujoco.rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
#     max_episode_steps=200
# )

# register(
#     id='HopperRandParams-v0',
#     entry_point='environments.mujoco.rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
#     max_episode_steps=200
# )


# # # 2D Navigation
# # # ----------------------------------------
# #
# register(
#     'PointEnv-v0',
#     entry_point='environments.navigation.point_robot:PointEnv',
#     kwargs={'goal_radius': 0.2,
#             'max_episode_steps': 100,
#             'goal_sampler': 'semi-circle'
#             },
#     max_episode_steps=100,
# )

# register(
#     'SparsePointEnv-v0',
#     entry_point='environments.navigation.point_robot:SparsePointEnv',
#     kwargs={'goal_radius': 0.2,
#             'max_episode_steps': 100,
#             'goal_sampler': 'semi-circle'
#             },
#     max_episode_steps=100,
# )

# #
# # # GridWorld
# # # ----------------------------------------

# register(
#     'GridNavi-v0',
#     entry_point='environments.navigation.gridworld:GridNavi',
#     kwargs={'num_cells': 5, 'num_steps': 15},
# )

# register(
#     'GridNaviNonStationary-v0',
#     entry_point='environments.navigation.gridworld_nonstationary:GridNaviNonStationary',
#     # kwargs={'num_cells': 5, 'num_steps': 15}, # comment this line to use the setting in gridworld_nonstationary.py
# )
