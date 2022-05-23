import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import matplotlib.pyplot as plt

import torch
from utils import helpers as utl
from utils.hidden_recoder import HiddenRecoder
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import progressbar

from scipy.stats import norm


class HalfCheetahWindNonstationary(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, ctx_in_state=False, normalize_ctx=True,  max_episode_steps=200):
        self.set_base_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1 # only consider wind on align to the direction of the robot

        self.curr_step = 0
        self.r_t = 0  # how many steps since last task change
        self.change_interval_base = 50
        self.has_change_interval = False

        self.ctx_in_state = ctx_in_state
        self.ctx_multiplier = 1 / 1000
        if torch.cuda.device_count() == 4:
            mujoco_env.MujocoEnv.__init__(
                            self, "/home/yufeng/Latent_Adaptive_RL/22_5_3_VariBAD/environments/mujoco/assets/half_cheetah_wind.xml", 5)
        elif torch.cuda.device_count() == 1:
            mujoco_env.MujocoEnv.__init__(
                            self, "/home/v-yuzheng/Latent_Adaptive_RL/22_5_3_VariBAD/environments/mujoco/assets/half_cheetah_wind.xml", 5)
        else:
            mujoco_env.MujocoEnv.__init__(
                self, "/Users/hector/Desktop/Latent_Adaptive_RL/22_5_3_VariBAD/environments/mujoco/assets/half_cheetah_wind.xml", 5)
        utils.EzPickle.__init__(self)

        # https://mujoco.readthedocs.io/en/latest/XMLreference.html
        self.model.opt.density = 10
        # wind: real(3), “0 0 0”
        # Velocity vector of the medium (i.e., wind).
        # This vector is subtracted from the 3D translational velocity of each body, and the result is used to compute viscous,
        # lift and drag forces acting on the body; recall Passive forces in the Computation chapter. 
        # The magnitude of these forces scales with the values of density.

        # density: real, “0”
        # Density of the medium, not to be confused with the geom density used to infer masses and inertias.
        # This parameter is used to simulate lift and drag forces, which scale quadratically with velocity.
        # In SI units the density of air is around 1.2 while the density of water is around 1000 depending on temperature.
        # Setting density to 0 disables lift and drag forces.

        # density and wind range can be set and visualized in mujoco app

        # the context dimension has been automatically accounted in _get_obs function
        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]


    def set_base_task(self, base_task):
        # set base task
        if isinstance(base_task, np.ndarray):
            base_task = base_task[0]
        self.wind_speed_base = base_task

    def sample_tasks(self, n_tasks):
        return [np.random.uniform(-15, 15) for _ in range(n_tasks)]

    def get_change_interval(self):
        self.change_interval = max(
            int(np.random.normal(self.change_interval_base, 10)), 10)
        self.has_change_interval = True
        
    def step(self, action):
        # self.step_curr_task += 1
        # self.step_count += 1
        self.r_t += 1
        self.curr_step += 1
        if not self.has_change_interval:
            self.get_change_interval()
        if self.r_t >= self.change_interval:
            # randomly choose a new task and set self.r_t to 0
            self.reset_task(None)

        self.sample_wind_speed()

        # xposbefore = self.sim.data.qpos[0]
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(action, self.frame_skip)
        # xposafter = self.sim.data.qpos[0]
        xposafter = self.get_body_com("torso")[0]
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        infos = dict(reward_forward=reward_run,
                    reward_ctrl=reward_ctrl,
                    curr_task=self.model.opt.wind[0],
                    curr_step=self.curr_step,
                    r_t=self.r_t,
                    curr_velocity=reward_run
        )
        return ob, reward, done, infos

    def _get_obs(self):
        if self.ctx_in_state:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                    np.array([self.model.opt.wind[0] * self.ctx_multiplier])
                ]
            )
        else:
            return np.concatenate(
                [
                    self.sim.data.qpos.flat[1:],
                    self.sim.data.qvel.flat,
                ]
            )

    def reset_task(self, base_task):
        if base_task is None:
            base_task = self.sample_tasks(1)[0]
        self.set_base_task(base_task)
        self.r_t = 0

    def get_task(self):
        return np.array([self.model.opt.wind[0]])

    def sample_wind_speed(self):
        self.model.opt.wind[0] = self.wind_speed_base + \
            np.random.normal(0, 0.02)

    def reset_model(self):
        self.step_curr_task = 0
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + \
            self.np_random.standard_normal(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

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
                plt.subplot(num_episodes, 4, i + 1)
                # (not plotting the last step because this gives weird artefacts)

                # location
                plt.plot(pos[i][:-1], range(len(pos[i][:-1])), 'k')
                plt.title('task: {}'.format(task), fontsize=15)
                plt.ylabel('steps (ep {})'.format(i), fontsize=15)
                if i == num_episodes - 1:
                    plt.xlabel('position', fontsize=15)
                # else:
                #     plt.xticks([])
                plt.xlim(min_x - 0.05 * span, max_x + 0.05 * span)
                plt.plot([0, 0], [200, 200], 'b--', alpha=0.2)



                # velocity & wind_speed (task)
                plt.subplot(num_episodes, 4, i + 1 + num_episodes)
                plt.plot(velocity_rec[i], range(len(velocity_rec[i])), 'k')
                if i == num_episodes - 1:
                    plt.xlabel('velocity', fontsize=15)


                plt.subplot(num_episodes, 4, i + 2 + num_episodes)
                plt.plot(episode_tasks[i], range(len(episode_tasks[i])), 'r')
                if i == num_episodes - 1:
                    plt.xlabel('wind_speed', fontsize=15)
                

                # unchanged length
                plt.subplot(num_episodes, 4, i + 3 + num_episodes)
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

        else:
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
                plt.subplot(num_episodes, 3, i + 1)
                # (not plotting the last step because this gives weird artefacts)

                # location
                plt.plot(pos[i][:-1], range(len(pos[i][:-1])), 'k')
                plt.title('task: {}'.format(task), fontsize=15)
                plt.ylabel('steps (ep {})'.format(i), fontsize=15)
                if i == num_episodes - 1:
                    plt.xlabel('position', fontsize=15)
                # else:
                #     plt.xticks([])
                plt.xlim(min_x - 0.05 * span, max_x + 0.05 * span)
                plt.plot([0, 0], [200, 200], 'b--', alpha=0.2)



                # velocity
                plt.subplot(num_episodes, 3, i + 1 + num_episodes)
                plt.plot(velocity_rec[i], range(len(velocity_rec[i])), 'k')
                
                if i == num_episodes - 1:
                    plt.xlabel('velocity', fontsize=15)


                # wind_speed (task)
                plt.subplot(num_episodes, 3, i + 2 + num_episodes)
                plt.plot(episode_tasks[i], range(len(episode_tasks[i])), 'r')
                if i == num_episodes - 1:
                    plt.xlabel('wind_speed', fontsize=15)


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
