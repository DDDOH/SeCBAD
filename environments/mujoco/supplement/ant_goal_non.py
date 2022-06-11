import progressbar
from utils.get_prior import get_prior
from utils import helpers as utl

import numpy as np
import torch
import os

from environments.mujoco.ant import AntEnv

import matplotlib.pyplot as plt
from utils.hidden_recoder import HiddenRecoder

from adaptive_learner import AdaptiveLearner  # we will use its inference method

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class AntGoalNon(AntEnv):
    """
    在二维平面上的圆内选goal
    """

    def __init__(self, max_episode_steps=200, **kwargs):
        self.given_task = kwargs['given_task'] if 'given_task' in kwargs.keys(
        ) else False

        self._max_episode_steps = max_episode_steps
        self.task_dim = 2

        if self.given_task:
            # load from file
            self.traj_task, self.segment_len_mean, self.segment_len_std, self.segment_len_min = self.load_traj_task(
                kwargs['task_dir_name'], kwargs['iter_idx'])

            self.apply_traj_task()
        else:
            # when training
            self.segment_len_mean = 80
            self.segment_len_std = 20
            self.segment_len_min = 10
            self.traj_task = self.sample_traj_task()

            self.apply_traj_task()

            # when first evaluate
            if 'task_dir_name' in kwargs.keys():
                if kwargs['task_dir_name'] is not None:
                    # sample and save traj_task for each evaluation step
                    self.sample_eval_traj_task(
                        kwargs['task_dir_name'], kwargs['vis_training_step'])

        self.p_G = None

        super(AntGoalNon, self).__init__()

    def load_traj_task(self, task_dir_name, iter_idx):
        # load npy file in task_dir_name
        task_dict = np.load(task_dir_name, allow_pickle=True).item()
        # task_dict is a dict of dict
        # task_dict key is training_step, value is traj_task
        # traj_task key is reset point, value is the new task value
        print('load_traj_task from {}'.format(task_dir_name))
        traj_task = task_dict['traj_task'][iter_idx]
        segment_len_mean = task_dict['segment_len_mean']
        segment_len_std = task_dict['segment_len_std']
        segment_len_min = task_dict['segment_len_min']
        return traj_task, segment_len_mean, segment_len_std, segment_len_min

    def apply_traj_task(self):
        # how to use traj_task
        self.segment_id = 0
        self.r_t = 0
        self.curr_step = 0
        self.traj_task_ls = list(self.traj_task.values())
        self.next_reset_step = list(self.traj_task.keys())[0]

    def sample_eval_traj_task(self, task_dir_name, vis_training_step):
        # sample and save traj_task for each evaluation step

        # check there's no file existing first
        if os.path.exists(task_dir_name):
            print('eval_traj_task already exists, do nothing')
        else:
            print('eval_traj_task not exists, create one and save at {}'.format(
                task_dir_name))
            task_dict = {'segment_len_mean': self.segment_len_mean,
                         'segment_len_std': self.segment_len_std,
                         'segment_len_min': self.segment_len_min,
                         'traj_task': {}}
            for training_step in vis_training_step:
                # sample traj_task
                task_dict['traj_task'][training_step] = self.sample_traj_task()
            # save task_dict
            np.save(task_dir_name, task_dict)

    def sample_traj_task(self):
        # sample one traj_task
        # traj_task is a dict
        # traj_task key is reset point, value is the new task value
        traj_task = {0: self.sample_new_task()}
        curr_step = 0

        while curr_step < self._max_episode_steps:
            next_segment_len = int(max(np.random.normal(
                self.segment_len_mean, self.segment_len_std), self.segment_len_min))
            new_task = self.sample_new_task()
            curr_step += next_segment_len
            traj_task[curr_step] = new_task

        return traj_task

    def sample_new_task(self):
        # sample a new task value
        # return np.random.choice([-1.0, 1.0])
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, 5)
        return [np.sin(angle) * radius, np.cos(angle) * radius]

    def get_p_G(self):
        if self.p_G is None:
            self.p_G_mat = get_prior(self.segment_len_mean, self.segment_len_std,
                                     self.segment_len_min, self._max_episode_steps, samples=10000)

            def p_G(G_t, G_t_minus_1):
                assert (G_t - G_t_minus_1 == 1) or (G_t == 1)
                return self.p_G_mat[G_t, G_t_minus_1]

            self.p_G = p_G

        return self.p_G

    def step(self, action):
        if self.curr_step == self.next_reset_step:
            self.r_t = -1  # add 1 and becomes 0 later.
            # 5_26
            if self.segment_id + 1 == len(self.traj_task_ls):
                # add 10 only for safty
                self.next_reset_step = self._max_episode_steps + 10
            else:
                self.next_reset_step = list(self.traj_task.keys())[
                    self.segment_id+1]
            self.set_task(self.traj_task_ls[self.segment_id])
            self.segment_id += 1

        self.goal_location = self.goal_location_base + np.random.normal(0, 0.2, size=2)

        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso")) 

        location_reward = - \
            np.linalg.norm(xposafter[:2] - self.goal_location)
        ctrl_cost = 0.05 * np.sum(np.square(action))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

        observation = self._get_obs()
        reward = location_reward - ctrl_cost - contact_cost
        done = False
        self.curr_step += 1
        self.r_t += 1

        infos = dict(
            location_reward=location_reward,
            reward_ctrl=-ctrl_cost,
            task=self.get_task(),
            curr_target=xposafter[:2],
            r_t=self.r_t)
        return observation, reward, done, infos

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_location_base = task

    def get_task(self):
        return np.array(self.goal_location)

    def reset_task(self, task=None):
        # 5_26
        if not self.given_task:
            # sample new traj task when not given task
            self.traj_task = self.sample_traj_task()
        self.apply_traj_task()
        return 0

    # 5_26 newly added, copied from ant_goal.py
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    @ staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            **kwargs,
                            ):

        # args.learner_type == varibad, sacbad, oracle_truncate

        # num_episodes = args.max_rollouts_per_task
        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

        # --- initialise things we want to keep track of ---

        # episode_prev_obs = [[] for _ in range(num_episodes)]
        # episode_next_obs = [[] for _ in range(num_episodes)]
        # episode_actions = [[] for _ in range(num_episodes)]
        # episode_rewards = [[] for _ in range(num_episodes)]

        prev_obs = []
        next_obs = []
        actions = []
        rewards = []

        # episode_returns = []
        # episode_lengths = []

        # episode_tasks = []
        tasks = []
        location_rec = []

        if encoder is not None:
            # episode_latent_samples = [[] for _ in range(num_episodes)]
            # episode_latent_means = [[] for _ in range(num_episodes)]
            # episode_latent_logvars = [[] for _ in range(num_episodes)]

            latent_samples = []
            latent_means = []
            latent_logvars = []
        else:
            curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
            # episode_latent_samples = episode_latent_means = episode_latent_logvars = None
            latent_samples = latent_means = latent_logvars = None

        # (re)set environment
        # env.reset_task()
        # 5_24 why the code below can not make experiment with different seed having the same task at the same iter_idx?
        # modify unwrapped_env won't change env?
        # unwrapped_env.reset(seed=iter_idx+42)
        state, belief, task, _ = utl.reset_env(env, args)
        start_state = state.clone()

        # if hasattr(args, 'hidden_size'):
        #     hidden_state = torch.zeros((1, args.hidden_size)).to(device)
        # else:
        #     hidden_state = None

        # keep track of what task we're in and the position of the cheetah
        # pos = []
        # start_pos = unwrapped_env.get_body_com("torso")[0].copy()

    # for episode_idx in range(num_episodes):

        if args.learner_type == 'sacbad':
            hidden_rec = HiddenRecoder(encoder)
            p_G_t_dist_rec = []

        # curr_rollout_rew = []
        # pos.append(start_pos)

        # episode_tasks.append([])
        # location_rec.append([])

        if encoder is not None:
            # if episode_idx == 0:
            # reset to prior
            if args.learner_type == 'sacbad':
                curr_latent_sample, curr_latent_mean, curr_latent_logvar = hidden_rec.encoder_init(
                    0)
            else:
                curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(
                    1)
                curr_latent_sample = curr_latent_sample[0].to(device)
                curr_latent_mean = curr_latent_mean[0].to(device)
                curr_latent_logvar = curr_latent_logvar[0].to(device)

            # episode_latent_samples[episode_idx].append(
            #     curr_latent_sample[0].clone())
            # episode_latent_means[episode_idx].append(
            #     curr_latent_mean[0].clone())
            # episode_latent_logvars[episode_idx].append(
            #     curr_latent_logvar[0].clone())

            latent_samples.append(curr_latent_sample[0].clone())
            latent_means.append(curr_latent_mean[0].clone())
            latent_logvars.append(curr_latent_logvar[0].clone())

        if args.learner_type == 'sacbad':
            # G_t_dist = {1: 1}
            p_G_t_dist = {1: 1}
            best_unchange_length_rec = []

        iterator = progressbar.progressbar(
            range(1, env._max_episode_steps + 1), redirect_stdout=True) if args.learner_type == 'sacbad' else range(1, env._max_episode_steps + 1)

        for step_idx in iterator:

            # if step_idx == 1:
            #     episode_prev_obs[episode_idx].append(start_state.clone())
            # else:
            #     episode_prev_obs[episode_idx].append(state.clone())

            if step_idx == 1:
                prev_obs.append(start_state.clone())
            else:
                prev_obs.append(state.clone())

            # act
            latent = utl.get_latent_for_policy(args,
                                               latent_sample=curr_latent_sample,
                                               latent_mean=curr_latent_mean,
                                               latent_logvar=curr_latent_logvar)
            _, action = policy.act(
                state=state.view(-1), latent=latent, belief=belief, task=task, deterministic=True)

            (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(
                env, action, args)
            state = state.reshape((1, -1)).float().to(device)

            # infos will not passed to agent
            # episode_tasks[-1].append(info[0]['task'])
            tasks.append(info[0]['task'])
            # location_rec[-1].append(info[0]['curr_target'])
            location_rec.append(info[0]['curr_target'])

            # keep track of position
            # pos[episode_idx].append(
            #     unwrapped_env.get_body_com("torso")[0].copy())
            # pos.append(
            #     unwrapped_env.get_body_com("torso")[0].copy())

            if encoder is not None:
                # update task embedding
                if args.learner_type == 'varibad':
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        action.reshape(
                            1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        hidden_state, return_prior=False)

                elif args.learner_type == 'sacbad':
                    # 5_23 use evaluate function in adaptive_learner.py
                    state_decoder = kwargs['state_decoder']
                    reward_decoder = kwargs['reward_decoder']
                    # prev_state = episode_prev_obs[episode_idx][-1]
                    prev_state = prev_obs[-1]

                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, best_unchange_length, p_G_t_dist = AdaptiveLearner.inference(
                        hidden_rec, prev_state, action, state, rew, step_idx, reward_decoder, state_decoder, p_G=unwrapped_env.get_p_G(), p_G_t_dist=p_G_t_dist)
                    best_unchange_length_rec.append(best_unchange_length)
                    p_G_t_dist_rec.append(p_G_t_dist)

                elif args.learner_type == 'oracle_truncate':
                    # 5_23 use real reset point
                    raise NotImplemented

                # episode_latent_samples[episode_idx].append(
                #     curr_latent_sample[0].clone())
                # episode_latent_means[episode_idx].append(
                #     curr_latent_mean[0].clone())
                # episode_latent_logvars[episode_idx].append(
                #     curr_latent_logvar[0].clone())

                latent_samples.append(curr_latent_sample[0].clone().to(device))
                latent_means.append(curr_latent_mean[0].clone().to(device))
                latent_logvars.append(curr_latent_logvar[0].clone().to(device))

            # episode_next_obs[episode_idx].append(state.clone())
            # episode_rewards[episode_idx].append(rew.clone())
            # episode_actions[episode_idx].append(
            #     action.reshape(1, -1).clone())

            next_obs.append(state.clone())
            rewards.append(rew.clone()[0][0])
            actions.append(action.reshape(1, -1).clone())

            if info[0]['done_mdp'] and not done:
                start_state = info[0]['start_state']
                start_state = torch.from_numpy(
                    start_state).reshape((1, -1)).float().to(device)
                # start_pos = unwrapped_env.get_body_com("torso")[0].copy()
                break

        # DONE 5_23 record all data for plot
        # DONE 5_23 remove num_episodes
        if args.learner_type == 'sacbad':
            p_G_t_dist_mat = np.zeros(
                (len(p_G_t_dist_rec)+1, len(p_G_t_dist_rec)+1))
            for i in range(len(p_G_t_dist_rec)):
                p_G_t_dist_mat[i, :i +
                               2] = list(p_G_t_dist_rec[i].values())[::-1]

        vis_dict = {'location_rec': location_rec,
                    'tasks': tasks,
                    'best_unchange_length_rec': best_unchange_length_rec if args.learner_type == 'sacbad' else None,
                    'p_G_t_dist_rec': p_G_t_dist_rec if args.learner_type == 'sacbad' else None,
                    'p_G_t_dist_mat': p_G_t_dist_mat if args.learner_type == 'sacbad' else None,
                    }

        np.save('{}/{}_vis_data.npy'.format(image_folder, iter_idx), vis_dict)

        # plot the movement of the half-cheetah
        # plt.figure(figsize=(7, 4 * num_episodes))

        # min_x = min([min(p) for p in pos])
        # max_x = max([max(p) for p in pos])
        # min_x = min(vis_dict['pos'])
        # max_x = max(vis_dict['pos'])
        # span = max_x - min_x
        # for i in range(num_episodes):

        # what to plot
        # location_x vs location_y
        # target_x vs target_y
        # location_x vs step, target_x vs step
        # location_y vs step, target_y vs step

        tasks = np.array(vis_dict['tasks'])
        location_rec = np.array(vis_dict['location_rec'])

        min_x = min(np.min(tasks[:, 0]), np.min(location_rec[:, 0])) - 1
        max_x = max(np.max(tasks[:, 0]), np.max(location_rec[:, 0])) + 1

        min_y = min(np.min(tasks[:, 1]), np.min(location_rec[:, 1])) - 1
        max_y = max(np.max(tasks[:, 1]), np.max(location_rec[:, 1])) + 1

        num_subplot = 4 + (args.learner_type == 'sacbad')
        plt.figure(figsize=(3 * num_subplot, 4))
        plt.subplot(1, num_subplot, 1)
        plt.scatter(tasks[:, 0], tasks[:, 1], c=np.arange(len(tasks)))
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.title('tasks')

        plt.subplot(1, num_subplot, 2)
        plt.scatter(location_rec[:, 0], location_rec[:,
                    1], c=np.arange(len(location_rec)))
        plt.title('agent location')
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.colorbar()

        plt.subplot(1, num_subplot, 3)
        plt.plot(tasks[:, 0], range(len(tasks)), c='r')
        plt.plot(location_rec[:, 0], range(len(tasks)), c='k')
        plt.xlim(min_x, max_x)
        plt.title('location & task x')

        plt.subplot(1, num_subplot, 4)
        plt.plot(tasks[:, 1], range(len(tasks)), c='r')
        plt.plot(location_rec[:, 1], range(len(tasks)), c='k')
        plt.xlim(min_y, max_y)
        plt.title('location & task y')

        if args.learner_type == 'sacbad':
            plt.subplot(1, num_subplot, 5)
            plt.plot(vis_dict['best_unchange_length_rec'], range(
                len(vis_dict['best_unchange_length_rec'])), 'k')
            plt.xlabel('G_t', fontsize=15)

            # plt.subplot(1, num_subplot, 4)
            # p_G_t_dist

        plt.tight_layout()
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        plt.close('all')

        if args.learner_type == 'sacbad':
            plt.figure()
            plt.imshow(vis_dict['p_G_t_dist_mat'])
            plt.savefig('{}/{}_p_G'.format(image_folder, iter_idx))
            plt.colorbar()
            # 让 vis_dict['p_G_t_dist_mat'] 每一行 除以 这一行的最大值
            # plt.imshow((vis_dict['p_G_t_dist_mat'].T/np.max(vis_dict['p_G_t_dist_mat'],axis=1)).T)
            plt.close('all')


        # 5_23 plot G_t triangle

        # if image_folder is not None:
        #     plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        #     plt.close()
        # else:
        #     plt.show()

        # if not return_pos:
        #     return episode_latent_means, episode_latent_logvars, \
        #         episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
        #         episode_returns, None
        # else:
        #     return episode_latent_means, episode_latent_logvars, \
        #         episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
        #         episode_returns, pos, None

        if not return_pos:
            return latent_means, latent_logvars, \
                next_obs, next_obs, actions, rewards, None
        else:
            return latent_means, latent_logvars, \
                next_obs, next_obs, actions, rewards, pos, None

        