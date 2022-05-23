from inspect import unwrap
import random

import numpy as np

from .half_cheetah import HalfCheetahEnv


import progressbar
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import helpers as utl
from utils.hidden_recoder import HiddenRecoder
from scipy.stats import norm

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HalfCheetahVelEnvNonstationary(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py

    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each
    time step a reward composed of a control cost and a penalty equal to the
    difference between its current velocity and the target velocity. The tasks
    are generated by sampling the target velocities from the uniform
    distribution on [0, 2].

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """

    # max_episode_steps is 200 originally
    def __init__(self, max_episode_steps):
        self.set_base_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1

        self.curr_step = 0
        self.r_t = 0  # how many steps since last task change
        self.change_interval_base = 60
        self.has_change_interval = False

        super(HalfCheetahVelEnvNonstationary, self).__init__()

    def get_change_interval(self):
        self.change_interval = max(
            int(np.random.normal(self.change_interval_base, 20)), 10)
        self.has_change_interval = True

    def step(self, action):
        self.r_t += 1
        self.curr_step += 1
        if not self.has_change_interval:
            self.get_change_interval()
        if self.r_t >= self.change_interval:
            # randomly choose a new task and set self.r_t to 0
            self.reset_task(None)

        self.sample_goal_velocity()

        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self.goal_velocity)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self.get_task(),
                     curr_task=self.goal_velocity,
                     curr_step=self.curr_step,
                     r_t=self.r_t,
                     curr_velocity=forward_vel,
                     ctrl_reward=-ctrl_cost,
                     vel_cost=forward_reward)
        return observation, reward, done, infos

    def sample_goal_velocity(self):
        # set task
        self.goal_velocity = self.goal_velocity_base + \
            np.random.normal(0, 0.12)

    def set_base_task(self, base_task):
        # set base task
        if isinstance(base_task, np.ndarray):
            base_task = base_task[0]
        self.goal_velocity_base = base_task

    def get_task(self):
        return np.array([self.goal_velocity])

    def sample_tasks(self, n_tasks):
        return [random.uniform(0.0, 3.0) for _ in range(n_tasks)]

    def reset_task(self, base_task):
        if base_task is None:
            base_task = self.sample_tasks(1)[0]
        self.set_base_task(base_task)
        self.r_t = 0
        # self.reset()

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

            # if hasattr(args, 'hidden_size'):
            #     hidden_state = torch.zeros((1, args.hidden_size)).to(device)
            # else:
            #     hidden_state = None

            # keep track of what task we're in and the position of the cheetah
            pos = [[] for _ in range(args.max_rollouts_per_task)]
            start_pos = unwrapped_env.get_body_com("torso")[0].copy()

            episode_tasks = []
            velocity_rec = []

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
                    velocity_rec[-1].append(info[0]['curr_velocity'])

                    # keep track of position
                    pos[episode_idx].append(
                        unwrapped_env.get_body_com("torso")[0].copy())

                    # 5_17 store g_G_t_dist

                    if encoder is not None:
                        # update task embedding
                        # curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = encoder(
                        #     action.reshape(
                        #         1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                        #     hidden_state, return_prior=False)

                        # when step_idx == 4, we have done the 4-th env step
                        # for the 5-th env step,
                        # we may use zero hidden_state (reset after 4 up to 4) (after 4-rd env step, the task switch to a new one)
                        # or latent & hidden_state reset after step 3 up to step 4
                        # or latent & hidden_state reset after step 2 up to step 4
                        # or latent & hidden_state reset after step 1 up to step 4
                        # or latent & hidden_state reset after step 0 up to step 4

                        # select one of the above case, use that as the hidden_state to take action in 5-th env step

                        # 旧的code相当于始终用 best_reset_after = 0
                        # the high in randint is exclusive

                        # choose using Bayesian inference
                        hidden_rec.encoder_step(action, state, rew)

                        g_G_t_dist = {}
                        # step_idx is still 4
                        for reset_after in range(step_idx):
                            # reset after in 0,1,2,3
                            # i = 5,4,3,2
                            # k = 4,3,2,1

                            i = step_idx + 1 - reset_after
                            k = step_idx - reset_after

                            # i in 5, 4, 3, 2

                            if (reset_after == 1) and (step_idx == 4):
                                a = 1

                            # assume reset_after is 1
                            # second term:

                            # get latent & hidden reset after 1 up to 4: second_term_before_cond
                            # and latent & hidden reset after 1 up to 3: second_term_after_cond
                            # and latent & hidden reset after 1 up to 2: second_term_after_cond
                            # (and latent & hidden reset after 1 up to 1?)

                            # print('before cond: reset_after {} up tp {}'.format(
                            #     reset_after, step_idx))
                            # print('after cond: reset_after {} up tp {}'.format(
                            #     reset_after, [j for j in range(reset_after, step_idx)]))
                            # print('i = {}'.format(i))

                            # second term is E_{ q(c|tau_{t-k:t-1}) } [ p(s_t, r_t-1 | s_t-1, a_t-1, c) ]

                            # get q(c|tau_{t-k:t-1})
                            # 5_17 check with xiaoyu
                            latent_samples, latent_mean, latent_logvar = hidden_rec.get_record(
                                reset_after=reset_after, up_to=step_idx, label='latent')

                            # 5_17 double check the code
                            reward_mean = kwargs['reward_decoder'](
                                latent_state=latent_samples, next_state=state, prev_state=episode_prev_obs[episode_idx][-1], actions=action)
                            state_mean = kwargs['state_decoder'](
                                latent_state=latent_samples, state=episode_prev_obs[episode_idx][-1], actions=action)

                            second_term = norm.pdf(rew.cpu().item(), loc=reward_mean.item(), scale=1)
                            second_term *= np.prod(norm.pdf(state.squeeze(0).cpu(), loc=state_mean.squeeze(0).cpu(), scale=1))

                            # second_term_before_cond = hidden_rec.get_record(
                            #     reset_after=reset_after, up_to=step_idx, label='latent')

                            # second_term_after_cond = [hidden_rec.get_record(
                            #     reset_after=reset_after, up_to=j, label='latent') for j in range(reset_after, step_idx)]  # python range(inclusive, exclusive)

                            # second_term = get_2nd_term(
                            #     second_term_before_cond, second_term_after_cond)

                            third_term = p_G(G_t=i, G_t_minus_1=i-1)
                            g_G_t_dist[i] = p_G_t_dist[i-1] * \
                                second_term * third_term

                            # G_t_dist[i] = P_t_minus_1_dist[i - 1] *

                        # reset_after = 4
                        # i = 1
                        # k = 0

                        g_G_t_dist[1] = 0
                        for k in range(1, step_idx+1):
                            # g_G_t_dist[1] += p_G_t_dist[k] * \
                            #     get_2nd_term_i_1(hidden_rec.get_record(
                            #         reset_after=step_idx, up_to=step_idx, label='latent')) * p_G(G_t=1, G_t_minus_1=k)

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

                        # print('best_reset_after: {}'.format(best_reset_after))

                        # print('p_G_t_dist: {}'.format(p_G_t_dist))
                        # print('best reset after {}'.format(best_reset_after))

                        # curr_latent_sample, curr_latent_mean, curr_latent_logvar = hidden_rec.encoder_step(
                        #     action, state, rew, best_reset_after)

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
                        start_pos = unwrapped_env.get_body_com("torso")[0].copy()
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

            # plot the movement of the half-cheetah
            plt.figure(figsize=(10, 4 * num_episodes))
            min_x = min([min(p) for p in pos])
            max_x = max([max(p) for p in pos])
            span = max_x - min_x
            for i in range(num_episodes):
                plt.subplot(num_episodes, 3, i + 1)
                # (not plotting the last step because this gives weird artefacts)
                plt.plot(pos[i][:-1], range(len(pos[i][:-1])), 'k')
                plt.title('task: {}'.format(task), fontsize=15)
                plt.ylabel('steps (ep {})'.format(i), fontsize=15)
                if i == num_episodes - 1:
                    plt.xlabel('position', fontsize=15)
                # else:
                #     plt.xticks([])
                plt.xlim(min_x - 0.05 * span, max_x + 0.05 * span)
                plt.plot([0, 0], [200, 200], 'b--', alpha=0.2)

                plt.subplot(num_episodes, 3, i + 1 + num_episodes)
                plt.plot(velocity_rec[i], range(len(velocity_rec[i])), 'k')
                plt.plot(episode_tasks[i], range(len(episode_tasks[i])), 'r')
                if i == num_episodes - 1:
                    plt.xlabel('velocity', fontsize=15)

                plt.subplot(num_episodes, 3, i + 2 + num_episodes)
                plt.plot(best_unchange_length_rec, range(
                    len(best_unchange_length_rec)), 'g')
                if i == num_episodes - 1:
                    plt.xlabel('unchanged length', fontsize=15)

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

            episode_tasks = []
            velocity_rec = []

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
            start_pos = unwrapped_env.get_body_com("torso")[0].copy()

            

            for episode_idx in range(num_episodes):

                curr_rollout_rew = []
                pos[episode_idx].append(start_pos)

                episode_tasks.append([])
                velocity_rec.append([])

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
                    velocity_rec[-1].append(info[0]['curr_velocity'])

                    # keep track of position
                    pos[episode_idx].append(unwrapped_env.get_body_com("torso")[0].copy())

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
                        start_pos = unwrapped_env.get_body_com("torso")[0].copy()
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

            # plot the movement of the half-cheetah
            plt.figure(figsize=(7, 4 * num_episodes))
            min_x = min([min(p) for p in pos])
            max_x = max([max(p) for p in pos])
            span = max_x - min_x
            for i in range(num_episodes):
                plt.subplot(num_episodes, 2, i + 1)
                # (not plotting the last step because this gives weird artefacts)
                plt.plot(pos[i][:-1], range(len(pos[i][:-1])), 'k')
                plt.title('task: {}'.format(task), fontsize=15)
                plt.ylabel('steps (ep {})'.format(i), fontsize=15)
                if i == num_episodes - 1:
                    plt.xlabel('position', fontsize=15)
                # else:
                #     plt.xticks([])
                plt.xlim(min_x - 0.05 * span, max_x + 0.05 * span)
                plt.plot([0, 0], [200, 200], 'b--', alpha=0.2)


                plt.subplot(num_episodes, 2, i + 1 + num_episodes)
                plt.plot(velocity_rec[i], range(len(velocity_rec[i])), 'k')
                plt.plot(episode_tasks[i], range(len(episode_tasks[i])), 'r')
                if i == num_episodes - 1:
                    plt.xlabel('velocity', fontsize=15)
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
            start_pos = unwrapped_env.get_body_com("torso")[0].copy()

            

            for episode_idx in range(num_episodes):

                curr_rollout_rew = []
                pos[episode_idx].append(start_pos)

                episode_tasks.append([])
                velocity_rec.append([])

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
                    velocity_rec[-1].append(info[0]['curr_velocity'])

                    # keep track of position
                    pos[episode_idx].append(unwrapped_env.get_body_com("torso")[0].copy())

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
                        start_pos = unwrapped_env.get_body_com("torso")[0].copy()
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

            # plot the movement of the half-cheetah
            plt.figure(figsize=(7, 4 * num_episodes))
            min_x = min([min(p) for p in pos])
            max_x = max([max(p) for p in pos])
            span = max_x - min_x
            for i in range(num_episodes):
                plt.subplot(num_episodes, 2, i + 1)
                # (not plotting the last step because this gives weird artefacts)
                plt.plot(pos[i][:-1], range(len(pos[i][:-1])), 'k')
                plt.title('task: {}'.format(task), fontsize=15)
                plt.ylabel('steps (ep {})'.format(i), fontsize=15)
                if i == num_episodes - 1:
                    plt.xlabel('position', fontsize=15)
                # else:
                #     plt.xticks([])
                plt.xlim(min_x - 0.05 * span, max_x + 0.05 * span)
                plt.plot([0, 0], [200, 200], 'b--', alpha=0.2)


                plt.subplot(num_episodes, 2, i + 1 + num_episodes)
                plt.plot(velocity_rec[i], range(len(velocity_rec[i])), 'k')
                plt.plot(episode_tasks[i], range(len(episode_tasks[i])), 'r')
                if i == num_episodes - 1:
                    plt.xlabel('velocity', fontsize=15)
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
