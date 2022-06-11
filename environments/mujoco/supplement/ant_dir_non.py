import progressbar
from utils.get_prior import get_prior
from utils import helpers as utl
import skvideo.io

import numpy as np
import torch
import os

from environments.mujoco.ant import AntEnv

import matplotlib.pyplot as plt
from utils.hidden_recoder import HiddenRecoder

from adaptive_learner import AdaptiveLearner  # we will use its inference method

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class AntDirNon(AntEnv):
    """
    Forward/backward ant direction environment

    set a direction on horizontal 2D plane as task
    与设定方向同向的速度，positive reward
    与设定方向垂直的速度，negative reward
    """

    def __init__(self, max_episode_steps=200, **kwargs):
        self.given_task = kwargs['given_task'] if 'given_task' in kwargs.keys(
        ) else False

        self._max_episode_steps = max_episode_steps
        self.task_dim = 1

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

        super(AntDirNon, self).__init__()

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
        angle_base = np.random.uniform(0, 2*np.pi)
        return angle_base

    def get_p_G(self, inaccurate_priori):
        if inaccurate_priori:
            # 5_25 how to set the following function properly?
            def p_G(G_t, G_t_minus_1):
                if G_t == self._max_episode_steps:
                    # 5_24 Check with Xiaoyu
                    return self.p_G_mat[G_t-1, G_t_minus_1-1]
                if G_t - G_t_minus_1 == 1:
                    return 1 - 1/self.segment_len_mean
                else:  # G_t = 1, G_t_minus_1 = k
                    return 1/self.segment_len_mean
        else:
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

        self.goal_angle = self.goal_angle_base + np.random.normal(0, 0.1)

        # self.goal_angle should stay within range [0, 2pi]
        if self.goal_angle > np.pi * 2:
            self.goal_angle -= np.pi * 2
        if self.goal_angle < 0:
            self.goal_angle += np.pi * 2

        self.goal_direction = np.array(
            [np.cos(self.goal_angle), np.sin(self.goal_angle)])

        torso_xyz_before = np.array(self.get_body_com("torso"))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        # torso_velocity[:2] / self.dt 是水平面上的 velocity
        agent_velocity = torso_velocity[:2] / self.dt
        forward_velocity = np.dot(agent_velocity, self.goal_direction)
        vertical_velocity = np.sqrt(
            agent_velocity[0]**2 + agent_velocity[1]**2 - forward_velocity**2)

        ctrl_cost = 0.05 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.data.cfrc_ext, -1, 1)))
        reward = 1 * forward_velocity - ctrl_cost - \
            contact_cost - 0.5 * vertical_velocity
        state = self.state_vector()

        observation = self._get_obs()
        done = False
        self.curr_step += 1
        self.r_t += 1

        curr_direction = self.get_curr_direction(agent_velocity)

        infos = dict(
            reward_forward=forward_velocity,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=None,
            torso_velocity=torso_velocity,
            task=self.get_task(),
            r_t=self.r_t,
            curr_location=torso_xyz_after[:2],
            curr_task=self.goal_angle,
            curr_direction=curr_direction,
            forward_velocity=forward_velocity,
        )

        return observation, reward, done, infos

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_angle_base = task

    def get_task(self):
        # 5_25 updated here
        return np.array([self.goal_angle])

    def get_curr_direction(self, agent_velocity):
        if (agent_velocity[0] > 0) & (agent_velocity[1] > 0):
            curr_direction = np.arctan(agent_velocity[1]/agent_velocity[0])
        elif (agent_velocity[0] > 0) & (agent_velocity[1] < 0):
            curr_direction = np.arctan(
                agent_velocity[1]/agent_velocity[0]) + 2 * np.pi
        elif (agent_velocity[0] < 0) & (agent_velocity[1] > 0):
            curr_direction = np.arctan(
                agent_velocity[1]/agent_velocity[0]) + np.pi
        elif (agent_velocity[0] < 0) & (agent_velocity[1] < 0):
            curr_direction = np.arctan(
                agent_velocity[1]/agent_velocity[0]) + np.pi
        return curr_direction

    def reset_task(self, task=None):
        # 5_26
        if not self.given_task:
            # sample new traj task when not given task
            self.traj_task = self.sample_traj_task()
        self.apply_traj_task()
        return 0

    @ staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            rendering=False,
                            **kwargs,
                            ):

        # TODO support load model
        # TODO render into video or use https://openai.github.io/mujoco-py/build/html/reference.html#mjviewer-3d-rendering

        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

        # --- initialise things we want to keep track of ---
        prev_obs = []
        next_obs = []
        actions = []
        rewards = []

        tasks = []
        location_rec = []
        curr_direction = []
        forward_velocity = []

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

        render_rec = []
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

            render_rec.append(env.render(mode='rgb_array'))

            # infos will not passed to agent
            # episode_tasks[-1].append(info[0]['task'])
            tasks.append(info[0]['task'])
            # location_rec[-1].append(info[0]['curr_target'])
            location_rec.append(info[0]['curr_location'])
            curr_direction.append(info[0]['curr_direction'])
            forward_velocity.append(info[0]['forward_velocity'])

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

                    inaccurate_priori = False if 'inaccurate_priori' not in kwargs else kwargs[
                        'inaccurate_priori']

                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, best_unchange_length, p_G_t_dist = AdaptiveLearner.inference(
                        hidden_rec, prev_state, action, state, rew, step_idx, reward_decoder, state_decoder, p_G=unwrapped_env.get_p_G(inaccurate_priori), p_G_t_dist=p_G_t_dist)
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
                    'curr_direction': curr_direction,
                    'forward_velocity': forward_velocity,
                    'best_unchange_length_rec': best_unchange_length_rec if args.learner_type == 'sacbad' else None,
                    'p_G_t_dist_rec': p_G_t_dist_rec if args.learner_type == 'sacbad' else None,
                    'p_G_t_dist_mat': p_G_t_dist_mat if args.learner_type == 'sacbad' else None,
                    }

        np.save('{}/{}_vis_data.npy'.format(image_folder, iter_idx), vis_dict)

        # plot the movement of ant
        # what to plot
        #
        # location x vs location y
        # task angle vs step, real angle vs step
        # vel along task angle vs step

        tasks = np.array(vis_dict['tasks'])
        location_rec = np.array(vis_dict['location_rec'])

        num_subplot = 3 + (args.learner_type == 'sacbad')
        plt.figure(figsize=(3 * num_subplot, 4))
        plt.subplot(1, num_subplot, 1)
        plt.scatter(location_rec[:, 0], location_rec[:,
                    1], c=np.arange(len(location_rec)))
        plt.title('location')
        plt.colorbar()

        plt.subplot(1, num_subplot, 2)
        plt.plot(vis_dict['curr_direction'], range(
            len(vis_dict['curr_direction'])), 'k')
        plt.plot(vis_dict['tasks'], range(len(vis_dict['tasks'])), 'r')
        plt.xlim(0, 2*np.pi)

        plt.subplot(1, num_subplot, 3)
        plt.plot(vis_dict['forward_velocity'], range(
            len(vis_dict['curr_direction'])), 'k')
        plt.title('velocity along task')

        if args.learner_type == 'sacbad':
            plt.subplot(1, num_subplot, 4)
            plt.plot(vis_dict['best_unchange_length_rec'], range(
                len(vis_dict['best_unchange_length_rec'])), 'k')
            plt.xlabel('G_t', fontsize=15)

            # plt.subplot(1, num_subplot, 4)
            # p_G_t_dist

        plt.tight_layout()
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        plt.close('all')

        render_rec = np.array(render_rec)
        skvideo.io.vwrite(
            '{}/{}_behaviour.mp4'.format(image_folder, iter_idx), render_rec)

        if args.learner_type == 'sacbad':
            plt.figure()
            plt.imshow(vis_dict['p_G_t_dist_mat'])
            plt.savefig('{}/{}_p_G'.format(image_folder, iter_idx))
            plt.colorbar()
            # 让 vis_dict['p_G_t_dist_mat'] 每一行 除以 这一行的最大值
            # plt.imshow((vis_dict['p_G_t_dist_mat'].T/np.max(vis_dict['p_G_t_dist_mat'],axis=1)).T)
            plt.close('all')

        if not return_pos:
            return latent_means, latent_logvars, \
                next_obs, next_obs, actions, rewards, None
        else:
            return latent_means, latent_logvars, \
                next_obs, next_obs, actions, rewards, pos, None
