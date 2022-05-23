import random

import numpy as np

from environments.mujoco.ant import AntEnv

import progressbar
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import helpers as utl
from utils.hidden_recoder import HiddenRecoder
from scipy.stats import norm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AntDir2DNonstationary(AntEnv):
    """
    Forward/backward ant direction environment

    set a direction on horizontal 2D plane as task
    与设定方向同向的速度，positive reward
    与设定方向垂直的速度，negative reward
    """

    def __init__(self, max_episode_steps=200):
        # self.set_task(self.sample_tasks())

        self.reset_task()
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1

        self.curr_step = 0
        self.r_t = 0  # how many steps since last task change
        self.change_interval_base = 60
        self.has_change_interval = False

        super(AntDir2DNonstationary, self).__init__()

    def get_change_interval(self):
        self.change_interval = max(
            int(np.random.normal(self.change_interval_base, 20)), 10)
        self.has_change_interval = True

    def reset_task(self, angle_base=None):
        if angle_base is None:
            angle_base = self.sample_tasks()
        self.set_base_task(angle_base)
        self.r_t = 0
        # 5_18 add this line to other nonstationary environments self.has_change_interval = False
        self.has_change_interval = False
        # self.reset()

    def set_base_task(self, angle_base):
        # set base task
        # if isinstance(base_task, np.ndarray):
        #     base_task = base_task[0]
        self.angle_base = angle_base

    def sample_tasks(self):
        # sample base task
        angle_base = np.random.uniform(0, 2*np.pi)
        return angle_base

    # def set_task(self, task):
    #     # if isinstance(task, np.ndarray):
    #     #     task = task[0]
    #     self.goal_angle = task

    def get_task(self):
        return np.array([self.goal_angle])

    def step(self, action):
        self.r_t += 1
        self.curr_step += 1
        if not self.has_change_interval:
            self.get_change_interval()
        if self.r_t >= self.change_interval:
            # randomly choose a new task and set self.r_t to 0
            self.reset_task(None)

        self.goal_angle = self.angle_base + np.random.normal(0, 0.02)
        self.goal_direction = np.array([np.cos(self.goal_angle), np.sin(self.goal_angle)])

        torso_xyz_before = np.array(self.get_body_com("torso"))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        # torso_velocity[:2] / self.dt 是水平面上的 velocity
        agent_velocity = torso_velocity[:2] / self.dt
        forward_velocity = np.dot(agent_velocity, self.goal_direction)
        vertical_velocity = np.sqrt( agent_velocity[0]**2 + agent_velocity[1]**2 - forward_velocity**2)
        

        # ctrl_cost = .5 * np.square(action).sum()
        # contact_cost = 0.5 * 1e-3 * np.sum(
        #     np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        # survive_reward = 1.0
        # reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        # state = self.state_vector()
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        # done = not notdone # the traj_len for this may not be 200!!!
        # ob = self._get_obs()
        
        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        reward = 1 * forward_velocity - ctrl_cost - contact_cost - 0.5 * vertical_velocity
        state = self.state_vector()
        done = False
        ob = self._get_obs()

        curr_direction = self.get_curr_direction(agent_velocity)

        return ob, reward, done, dict(
            reward_forward=forward_velocity,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=None,
            torso_velocity=torso_velocity,
            task=self.get_task(),
            r_t=self.r_t,
            curr_task=self.goal_angle,
            curr_direction=curr_direction,
            forward_velocity=forward_velocity,
        )

    def get_curr_direction(self, agent_velocity):
        if (agent_velocity[0] > 0) & (agent_velocity[1] > 0):
            curr_direction = np.arctan(agent_velocity[1]/agent_velocity[0])
        elif (agent_velocity[0] > 0) & (agent_velocity[1] < 0):
            curr_direction = np.arctan(agent_velocity[1]/agent_velocity[0]) + 2 * np.pi
        elif (agent_velocity[0] < 0) & (agent_velocity[1] > 0):
            curr_direction = np.arctan(agent_velocity[1]/agent_velocity[0]) + np.pi
        elif (agent_velocity[0] < 0) & (agent_velocity[1] < 0):
            curr_direction = np.arctan(agent_velocity[1]/agent_velocity[0]) + np.pi
        return curr_direction

    @staticmethod
    def visualise_behaviour(env,
                            args,
                            policy,
                            iter_idx,
                            encoder=None,
                            image_folder=None,
                            return_pos=False,
                            **kwargs,
                            ):


        if args.learner_type == 'adaptive':

            num_episodes = args.max_rollouts_per_task
            unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

            # --- initialise things we want to keep track of ---

            episode_prev_obs = [[] for _ in range(num_episodes)]
            episode_next_obs = [[] for _ in range(num_episodes)]
            episode_actions = [[] for _ in range(num_episodes)]
            episode_rewards = [[] for _ in range(num_episodes)]

            episode_returns = []
            episode_lengths = []

            if encoder is not None:
                episode_latent_samples = [[] for _ in range(num_episodes)]
                episode_latent_means = [[] for _ in range(num_episodes)]
                episode_latent_logvars = [[] for _ in range(num_episodes)]
            else:
                curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
                episode_latent_samples = episode_latent_means = episode_latent_logvars = None

            # --- roll out policy ---

            # (re)set environment
            env.reset_task()
            state, belief, task, _ = utl.reset_env(env, args)
            start_state = state.clone()

            pos = [[] for _ in range(args.max_rollouts_per_task)]
            start_pos = unwrapped_env.get_body_com("torso")[:2].copy()

            episode_tasks = []
            velocity_rec = []
            episode_direction = []

            def p_G(G_t, G_t_minus_1):
                # 5_17  update this function
                if G_t - G_t_minus_1 == 1:
                    return 1 - 1/unwrapped_env.change_interval_base
                else:
                    # G_t = 1, G_t_minus_1 = k
                    return 1/unwrapped_env.change_interval_base


            for episode_idx in range(num_episodes):

                hidden_rec = HiddenRecoder(encoder)

                curr_rollout_rew = []
                pos[episode_idx].append(start_pos)

                episode_tasks.append([])
                velocity_rec.append([])
                episode_direction.append([])

                if encoder is not None:
                    if episode_idx == 0:
                        # reset to prior
                        # curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(
                        #     1)

                        curr_latent_sample, curr_latent_mean, curr_latent_logvar = hidden_rec.encoder_init(
                            0)
                        if curr_latent_sample.dim() == 3:
                            curr_latent_sample = curr_latent_sample[0].to(device)
                            curr_latent_mean = curr_latent_mean[0].to(device)
                            curr_latent_logvar = curr_latent_logvar[0].to(device)
                        else:
                            curr_latent_sample = curr_latent_sample.to(device)
                            curr_latent_mean = curr_latent_mean.to(device)
                            curr_latent_logvar = curr_latent_logvar.to(device)
                    episode_latent_samples[episode_idx].append(
                        curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(
                        curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(
                        curr_latent_logvar[0].clone())

                G_t_dist = {1: 1}
                p_G_t_dist = {1: 1}

                best_unchange_length_rec = []

                # for step_idx in range(1, env._max_episode_steps + 1):
                for step_idx in progressbar.progressbar(range(1, env._max_episode_steps + 1), redirect_stdout=True):
                    if step_idx == 1:
                        episode_prev_obs[episode_idx].append(start_state.clone())
                    else:
                        episode_prev_obs[episode_idx].append(state.clone())
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
                    episode_tasks[-1].append(info[0]['curr_task'])
                    velocity_rec[-1].append(info[0]['forward_velocity'])
                    episode_direction[-1].append(info[0]['curr_direction'])


                    # keep track of position
                    pos[episode_idx].append(
                        unwrapped_env.get_body_com("torso")[:2].copy())

                    # 5_17 store g_G_t_dist

                    if encoder is not None:
                        hidden_rec.encoder_step(action, state, rew)

                        g_G_t_dist = {}
                        # step_idx is still 4
                        for reset_after in range(step_idx):

                            i = step_idx + 1 - reset_after
                            k = step_idx - reset_after

                            # i in 5, 4, 3, 2

                            if (reset_after == 1) and (step_idx == 4):
                                a = 1

                            latent_samples, latent_mean, latent_logvar = hidden_rec.get_record(
                                reset_after=reset_after, up_to=step_idx, label='latent')

                            # 5_17 double check the code
                            reward_mean = kwargs['reward_decoder'](
                                latent_state=latent_samples, next_state=state, prev_state=episode_prev_obs[episode_idx][-1], actions=action)
                            state_mean = kwargs['state_decoder'](
                                latent_state=latent_samples, state=episode_prev_obs[episode_idx][-1], actions=action)

                            second_term = norm.pdf(rew.cpu().item(), loc=reward_mean.item(), scale=1)
                            second_term *= np.prod(norm.pdf(state.squeeze(0).cpu(), loc=state_mean.squeeze(0).cpu(), scale=1))


                            third_term = p_G(G_t=i, G_t_minus_1=i-1)
                            g_G_t_dist[i] = p_G_t_dist[i-1] * \
                                second_term * third_term


                        g_G_t_dist[1] = 0
                        for k in range(1, step_idx+1):

                            latent_samples, latent_mean, latent_logvar = hidden_rec.get_record(
                                    reset_after=step_idx, up_to=step_idx, label='latent')

                            reward_mean = kwargs['reward_decoder'](
                                latent_state=latent_samples, next_state=state, prev_state=episode_prev_obs[episode_idx][-1], actions=action)
                            state_mean = kwargs['state_decoder'](
                                latent_state=latent_samples, state=episode_prev_obs[episode_idx][-1], actions=action)

                            second_term = norm.pdf(rew.cpu().item(), loc=reward_mean.item(), scale=1)
                            second_term *= np.prod(norm.pdf(state.squeeze(0).cpu(), loc=state_mean.squeeze(0).cpu(), scale=1))

                            g_G_t_dist[1] += p_G_t_dist[k] * second_term * p_G(G_t=1, G_t_minus_1=k)


                        # get sum of g_G_t_dist
                        sum_g_G_t = sum(g_G_t_dist.values())
                        # divide each value of g_G_t_dist by sum_g_G_t
                        # use for next iteration
                        p_G_t_dist = {k: v / sum_g_G_t for k,
                                    v in g_G_t_dist.items()}

                        best_unchange_length = max(g_G_t_dist, key=g_G_t_dist.get)
                        best_reset_after = step_idx + 1 - best_unchange_length
                        best_unchange_length_rec.append(best_unchange_length)


                        curr_latent_sample, curr_latent_mean, curr_latent_logvar = hidden_rec.get_record(
                            reset_after=best_reset_after, up_to=step_idx, label='latent')
                        # print('reset_after: {}, up_to: {}'.format(best_reset_after, step_idx))

                        assert curr_latent_sample.dim() == 2
                        assert curr_latent_mean.dim() == 2
                        assert curr_latent_logvar.dim() == 2

                        episode_latent_samples[episode_idx].append(
                            curr_latent_sample[0].clone())
                        episode_latent_means[episode_idx].append(
                            curr_latent_mean[0].clone())
                        episode_latent_logvars[episode_idx].append(
                            curr_latent_logvar[0].clone())

                    episode_next_obs[episode_idx].append(state.clone())
                    episode_rewards[episode_idx].append(rew.clone())
                    episode_actions[episode_idx].append(
                        action.reshape(1, -1).clone())

                    if info[0]['done_mdp'] and not done:
                        start_state = info[0]['start_state']
                        start_state = torch.from_numpy(
                            start_state).reshape((1, -1)).float().to(device)
                        start_pos = unwrapped_env.get_body_com("torso")[:2].copy()
                        break

                episode_returns.append(sum(curr_rollout_rew))
                episode_lengths.append(step_idx)

            # clean up
            if encoder is not None:
                episode_latent_means = [torch.stack(
                    e) for e in episode_latent_means]
                episode_latent_logvars = [torch.stack(
                    e) for e in episode_latent_logvars]

            episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
            episode_next_obs = [torch.cat(e) for e in episode_next_obs]
            episode_actions = [torch.cat(e) for e in episode_actions]
            episode_rewards = [torch.cat(e) for e in episode_rewards]

            plt.figure(figsize=(15, 4 * num_episodes))
            pos = [np.array(p) for p in pos]

            min_x = min(min(p[:,0]) for p in pos)
            max_x = max(max(p[:,0]) for p in pos)
            min_y = min(min(p[:,1]) for p in pos)
            max_y = max(max(p[:,1]) for p in pos)
            span_x = max_x - min_x
            span_y = max_y - min_y
            for i in range(num_episodes):
                # position_x & pos_y
                plt.subplot(num_episodes, 4, i + 1)
                plt.scatter(pos[i][:,0], pos[i][:,1], c=np.arange(pos[i].shape[0]))
                plt.xlim(min_x - 0.05 * span_x, max_x + 0.05 * span_x)
                plt.ylim(min_y - 0.05 * span_y, max_y + 0.05 * span_y)
                plt.xlabel('pos_x')
                plt.ylabel('pos_y')
                plt.colorbar()

                # task angle & step
                # true angle & step
                plt.subplot(num_episodes, 4, i + 1 + num_episodes)
                plt.plot(episode_direction[i], range(len(episode_direction[i])), 'k')
                plt.plot(episode_tasks[i], range(len(episode_tasks[i])), 'r')
                plt.xlim(-0.1, 2*np.pi+0.1)
                plt.title('angle')

                # velocity along task angle
                plt.subplot(num_episodes, 4, i + 2 + num_episodes)
                plt.plot(velocity_rec[i], range(len(velocity_rec[i])), 'k')
                plt.title('velocity along task angle')

                # unchanged_length & step
                plt.subplot(num_episodes, 4, i + 3 + num_episodes)
                plt.plot(best_unchange_length_rec, range(
                    len(best_unchange_length_rec)), 'g')
            plt.tight_layout()
            if image_folder is not None:
                plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
                plt.close('all')
            else:
                plt.show()

            episode_tasks = [torch.tensor(episode_task).unsqueeze(
                0).T for episode_task in episode_tasks]

            if not return_pos:
                return episode_latent_means, episode_latent_logvars, \
                    episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                    episode_returns, episode_tasks
            else:
                return episode_latent_means, episode_latent_logvars, \
                    episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                    episode_returns, pos, episode_tasks

        elif args.learner_type == 'oracle_truncate':
            num_episodes = args.max_rollouts_per_task
            unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

            # --- initialise things we want to keep track of ---

            episode_prev_obs = [[] for _ in range(num_episodes)]
            episode_next_obs = [[] for _ in range(num_episodes)]
            episode_actions = [[] for _ in range(num_episodes)]
            episode_rewards = [[] for _ in range(num_episodes)]

            episode_returns = []
            episode_lengths = []

            

            if encoder is not None:
                episode_latent_samples = [[] for _ in range(num_episodes)]
                episode_latent_means = [[] for _ in range(num_episodes)]
                episode_latent_logvars = [[] for _ in range(num_episodes)]
            else:
                curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
                episode_latent_samples = episode_latent_means = episode_latent_logvars = None

            #                     # second term is E_{ q(c|tau_{t-k:t-1}) } [ p(s_t, r_t-1 | s_t-1, a_t-1, c) ]

            # (re)set environment
            env.reset_task()
            state, belief, task, _ = utl.reset_env(env, args)
            start_state = state.clone()

            # if hasattr(args, 'hidden_size'):
            #     hidden_state = torch.zeros((1, args.hidden_size)).to(device)
            # else:
            #     hidden_state = None

            # keep track of what task we're in and the position of the cheetah
            pos = [[] for _ in range(args.max_rollouts_per_task)]
            start_pos = unwrapped_env.get_body_com("torso")[:2].copy()

            episode_tasks = []
            velocity_rec = []
            episode_direction = []

            for episode_idx in range(num_episodes):

                curr_rollout_rew = []
                pos[episode_idx].append(start_pos)

                episode_tasks.append([])
                velocity_rec.append([])
                episode_direction.append([])

                if encoder is not None:
                    if episode_idx == 0:
                        # reset to prior
                        curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                        curr_latent_sample = curr_latent_sample[0].to(device)
                        curr_latent_mean = curr_latent_mean[0].to(device)
                        curr_latent_logvar = curr_latent_logvar[0].to(device)
                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                for step_idx in range(1, env._max_episode_steps + 1):

                    if step_idx == 1:
                        episode_prev_obs[episode_idx].append(start_state.clone())
                    else:
                        episode_prev_obs[episode_idx].append(state.clone())
                    # act
                    latent = utl.get_latent_for_policy(args,
                                                    latent_sample=curr_latent_sample,
                                                    latent_mean=curr_latent_mean,
                                                    latent_logvar=curr_latent_logvar)
                    _, action = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task, deterministic=True)

                    (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                    state = state.reshape((1, -1)).float().to(device)
                    r_t = torch.tensor([info[0]['r_t']]).to(device)
                    done = torch.tensor(done).to(device)
                    # assert done is False # done should be false (?)
                    #                     # if r_t is 0, reset hidden
                    # # see the training code of oracle_truncate

                    # r_t = torch.from_numpy(np.array([info['r_t'] for info in infos], dtype=int)).to(
                    #     device).float().view((-1, 1))  # r_t = 0 表示前一个循环 刚刚 reset 过 task

                    # done = torch.from_numpy(np.array(done, dtype=int)).to(
                    #     device).float().view((-1, 1))
                    # # create mask for episode ends
                    # masks_done = torch.FloatTensor(
                    #     [[0.0] if done_ else [1.0] for done_ in done]).to(device)
                    # bad_mask is true if episode ended because time limit was reached
                    # bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [
                    #                             1.0] for info in infos]).to(device)

        
                    # infos will not passed to agent
                    episode_tasks[-1].append(info[0]['curr_task'])
                    velocity_rec[-1].append(info[0]['forward_velocity'])
                    episode_direction[-1].append(info[0]['curr_direction'])

                    # keep track of position
                    pos[episode_idx].append(unwrapped_env.get_body_com("torso")[:2].copy())

                    if encoder is not None:
                        # update task embedding
                        with torch.no_grad():
                        # compute next embedding (for next loop and/or value prediction bootstrap)
                        # 这里 是在 前一个 state 上，用 action, 得到 rew_raw 以及 next_state，对应的是 a_t-1, r_t, s_t, 和 paper 里 figure 2 对应
                            curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = utl.update_encoding(
                                encoder=encoder,
                                next_obs=state,
                                action=action,
                                reward=rew,
                                done=done,
                                hidden_state=hidden_state,
                                r_t=r_t)
                        
                        # curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        #     action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        #     hidden_state, return_prior=False)

                        episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                        episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                        episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                    episode_next_obs[episode_idx].append(state.clone())
                    episode_rewards[episode_idx].append(rew.clone())
                    episode_actions[episode_idx].append(action.reshape(1, -1).clone())

                    if info[0]['done_mdp'] and not done:
                        start_state = info[0]['start_state']
                        start_state = torch.from_numpy(start_state).reshape((1, -1)).float().to(device)
                        start_pos = unwrapped_env.get_body_com("torso")[:2].copy()
                        break

                episode_returns.append(sum(curr_rollout_rew))
                episode_lengths.append(step_idx)

            # clean up
            if encoder is not None:
                episode_latent_means = [torch.stack(
                    e) for e in episode_latent_means]
                episode_latent_logvars = [torch.stack(
                    e) for e in episode_latent_logvars]

            episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
            episode_next_obs = [torch.cat(e) for e in episode_next_obs]
            episode_actions = [torch.cat(e) for e in episode_actions]
            episode_rewards = [torch.cat(e) for e in episode_rewards]

            plt.figure(figsize=(15, 4 * num_episodes))
            pos = [np.array(p) for p in pos]
            
            min_x = min(min(p[:,0]) for p in pos)
            max_x = max(max(p[:,0]) for p in pos)
            min_y = min(min(p[:,1]) for p in pos)
            max_y = max(max(p[:,1]) for p in pos)
            span_x = max_x - min_x
            span_y = max_y - min_y
            for i in range(num_episodes):
                # position_x & pos_y
                plt.subplot(num_episodes, 3, i + 1)
                plt.scatter(pos[i][:,0], pos[i][:,1], c=np.arange(pos[i].shape[0]))
                plt.xlim(min_x - 0.05 * span_x, max_x + 0.05 * span_x)
                plt.ylim(min_y - 0.05 * span_y, max_y + 0.05 * span_y)
                plt.xlabel('pos_x')
                plt.ylabel('pos_y')
                plt.colorbar()

                # task angle & step
                # true angle & step
                plt.subplot(num_episodes, 3, i + 1 + num_episodes)
                plt.plot(episode_direction[i], range(len(episode_direction[i])), 'k')
                plt.plot(episode_tasks[i], range(len(episode_tasks[i])), 'r')
                plt.xlim(-0.1, 2*np.pi+0.1)
                plt.title('angle')

                plt.subplot(num_episodes, 3, i + 2 + num_episodes)
                # velocity along task angle
                plt.subplot(num_episodes, 4, i + 2 + num_episodes)
                plt.plot(velocity_rec[i], range(len(velocity_rec[i])), 'k')
                plt.title('velocity along task angle')


            plt.tight_layout()
            if image_folder is not None:
                plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
                plt.close()
            else:
                plt.show()

            if not return_pos:
                return episode_latent_means, episode_latent_logvars, \
                    episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                    episode_returns, None
            else:
                return episode_latent_means, episode_latent_logvars, \
                    episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                    episode_returns, pos, None


        
        elif args.learner_type == 'varibad':
            num_episodes = args.max_rollouts_per_task
            unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

            # --- initialise things we want to keep track of ---

            episode_prev_obs = [[] for _ in range(num_episodes)]
            episode_next_obs = [[] for _ in range(num_episodes)]
            episode_actions = [[] for _ in range(num_episodes)]
            episode_rewards = [[] for _ in range(num_episodes)]

            episode_returns = []
            episode_lengths = []

            episode_tasks = []
            velocity_rec = []
            episode_direction = []

            if encoder is not None:
                episode_latent_samples = [[] for _ in range(num_episodes)]
                episode_latent_means = [[] for _ in range(num_episodes)]
                episode_latent_logvars = [[] for _ in range(num_episodes)]
            else:
                curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
                episode_latent_samples = episode_latent_means = episode_latent_logvars = None

            #                     # second term is E_{ q(c|tau_{t-k:t-1}) } [ p(s_t, r_t-1 | s_t-1, a_t-1, c) ]

            # (re)set environment
            env.reset_task()
            state, belief, task, _ = utl.reset_env(env, args)
            start_state = state.clone()

            # if hasattr(args, 'hidden_size'):
            #     hidden_state = torch.zeros((1, args.hidden_size)).to(device)
            # else:
            #     hidden_state = None

            # keep track of what task we're in and the position of the cheetah
            pos = [[] for _ in range(args.max_rollouts_per_task)]
            start_pos = unwrapped_env.get_body_com("torso")[:2].copy()

            

            for episode_idx in range(num_episodes):

                curr_rollout_rew = []
                pos[episode_idx].append(start_pos)

                episode_tasks.append([])
                velocity_rec.append([])
                episode_direction.append([])

                if encoder is not None:
                    if episode_idx == 0:
                        # reset to prior
                        curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder.prior(1)
                        curr_latent_sample = curr_latent_sample[0].to(device)
                        curr_latent_mean = curr_latent_mean[0].to(device)
                        curr_latent_logvar = curr_latent_logvar[0].to(device)
                    episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                    episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                    episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                for step_idx in range(1, env._max_episode_steps + 1):

                    if step_idx == 1:
                        episode_prev_obs[episode_idx].append(start_state.clone())
                    else:
                        episode_prev_obs[episode_idx].append(state.clone())
                    # act
                    latent = utl.get_latent_for_policy(args,
                                                    latent_sample=curr_latent_sample,
                                                    latent_mean=curr_latent_mean,
                                                    latent_logvar=curr_latent_logvar)
                    _, action = policy.act(state=state.view(-1), latent=latent, belief=belief, task=task, deterministic=True)

                    (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(env, action, args)
                    state = state.reshape((1, -1)).float().to(device)

                    # infos will not passed to agent
                    episode_tasks[-1].append(info[0]['curr_task'])
                    velocity_rec[-1].append(info[0]['forward_velocity'])
                    episode_direction[-1].append(info[0]['curr_direction'])

                    # keep track of position
                    pos[episode_idx].append(unwrapped_env.get_body_com("torso")[:2].copy())

                    if encoder is not None:
                        # update task embedding
                        curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                            action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                            hidden_state, return_prior=False)

                        episode_latent_samples[episode_idx].append(curr_latent_sample[0].clone())
                        episode_latent_means[episode_idx].append(curr_latent_mean[0].clone())
                        episode_latent_logvars[episode_idx].append(curr_latent_logvar[0].clone())

                    episode_next_obs[episode_idx].append(state.clone())
                    episode_rewards[episode_idx].append(rew.clone())
                    episode_actions[episode_idx].append(action.reshape(1, -1).clone())

                    if info[0]['done_mdp'] and not done:
                        start_state = info[0]['start_state']
                        start_state = torch.from_numpy(start_state).reshape((1, -1)).float().to(device)
                        start_pos = unwrapped_env.get_body_com("torso")[:2].copy()
                        break

                episode_returns.append(sum(curr_rollout_rew))
                episode_lengths.append(step_idx)

            # clean up
            if encoder is not None:
                episode_latent_means = [torch.stack(e) for e in episode_latent_means]
                episode_latent_logvars = [torch.stack(e) for e in episode_latent_logvars]

            episode_prev_obs = [torch.cat(e) for e in episode_prev_obs]
            episode_next_obs = [torch.cat(e) for e in episode_next_obs]
            episode_actions = [torch.cat(e) for e in episode_actions]
            episode_rewards = [torch.cat(e) for e in episode_rewards]

            plt.figure(figsize=(15, 4 * num_episodes))
            pos = [np.array(p) for p in pos]
            
            min_x = min(min(p[:,0]) for p in pos)
            max_x = max(max(p[:,0]) for p in pos)
            min_y = min(min(p[:,1]) for p in pos)
            max_y = max(max(p[:,1]) for p in pos)
            span_x = max_x - min_x
            span_y = max_y - min_y
            for i in range(num_episodes):
                # position_x & pos_y
                plt.subplot(num_episodes, 3, i + 1)
                plt.scatter(pos[i][:,0], pos[i][:,1], c=np.arange(pos[i].shape[0]))
                plt.xlim(min_x - 0.05 * span_x, max_x + 0.05 * span_x)
                plt.ylim(min_y - 0.05 * span_y, max_y + 0.05 * span_y)
                plt.xlabel('pos_x')
                plt.ylabel('pos_y')
                plt.colorbar()

                # task angle & step
                # true angle & step
                plt.subplot(num_episodes, 3, i + 1 + num_episodes)
                plt.plot(episode_direction[i], range(len(episode_direction[i])), 'k')
                plt.plot(episode_tasks[i], range(len(episode_tasks[i])), 'r')
                plt.xlim(-0.1, 2*np.pi+0.1)
                plt.title('angle')

                plt.subplot(num_episodes, 3, i + 2 + num_episodes)
                plt.plot(velocity_rec[i], range(len(velocity_rec[i])), 'k')
                plt.title('velocity along task angle')

            plt.tight_layout()
            if image_folder is not None:
                plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
                plt.close()
            else:
                plt.show()

            if not return_pos:
                return episode_latent_means, episode_latent_logvars, \
                    episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                    episode_returns, None
            else:
                return episode_latent_means, episode_latent_logvars, \
                    episode_prev_obs, episode_next_obs, episode_actions, episode_rewards, \
                    episode_returns, pos, None



# class AntDir2DEnv(AntDirEnv):
#     def sample_tasks(self, n_tasks):
#         # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
#         directions = np.array([random.gauss(mu=0, sigma=1) for _ in range(n_tasks * 2)]).reshape((n_tasks, 2))
#         directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
#         return directions


# class AntDirOracleEnv(AntDirEnv):
#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat[2:],
#             self.sim.data.qvel.flat,
#             [self.goal_direction],
#         ])


# class AntDir2DOracleEnv(AntDir2DEnv):
#     def _get_obs(self):
#         return np.concatenate([
#             self.sim.data.qpos.flat[2:],
#             self.sim.data.qvel.flat,
#             [self.goal_direction],
#         ])
