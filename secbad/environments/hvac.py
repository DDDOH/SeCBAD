import numpy as np
import matplotlib.pyplot as plt


from gym import spaces
import gym

if __name__ != '__main__':
    from ..utils import helpers as utl
    # device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

import torch


# TODO force hvac only have cooling functionality, when DAT > zone temperature or something else, raise a warrning

def positive_part(input):
    return (input + np.abs(input))/2


def negative_part(input):
    return (input - np.abs(input))/2


# 5_30 randomly choose new building with different parameters
# 5_30 consider using normalized data scale
class HVAC(gym.Env):
    def __init__(self, traj_len, id=None):
        super(HVAC, self).__init__()

        self.traj_len = traj_len
        self.id = id

        # set AHU parameters
        self.AHU_paras = {'a': 0.1, 'COP': 3, 'mu': 0.8}

        self.len_time_slot = 0.25

        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,))
        self.action_space = spaces.Box(
            low=0, high=100, shape=(2,))

        self.context_dim = 10

        self.max_DAS = 80

        # 6_7 just put something here, not sure if its the best way to do it
        self.init_DAS = self.max_DAS
        self.init_SPS = 100

        self.prev_SPS = self.init_SPS
        self.prev_DAS = self.init_DAS

    def step(self, action):
        """
        Execute one step in the environment.
        Should return: state, reward, done, info
        where info has to include a field 'task'.
        """
        # if action.ndim == 2:
        #     action = action[0]

        # TODO what to do when DAS > T_i?
        DAS_i, SPS_i = action[0] * 5 + \
            self.init_DAS, action[1] * 10 + self.init_SPS

        if SPS_i < 0.01:
            SPS_i = 0.01

        m_i = np.random.uniform(
            0.95, 1.05, size=self.n_building) * SPS_i / self.n_building

        # DAT is roughly equal to DAS, the random number is the control error of air handling unit
        DAT_i = DAS_i + np.random.normal(0, 3)

        T_s = DAT_i  # T_s is the supply air temperature of VFD, which is DAT of air handling unit

        if self.step_idx == 0:
            self.T_i = self.T_i_0

        self.T_i, RAT_i = self._get_T_t_plus_1(
            self.T_i, m_i, T_s, self.len_time_slot, self.OAT[self.step_idx])

        # TODO get MAT using RAT & OAT, use the equation in page 3 in ppt (may not be a good one)
        MAT_i = self.a * RAT_i + (1 - self.a) * self.OAT[self.step_idx]

        power_cost = self._get_cost_cooling_coil(m_i, T_s, MAT=MAT_i, COP=self.COP, lambda_t=self.electric_price[self.step_idx], len_time_slot=self.len_time_slot, eta=0.1) + \
            self._get_cost_supply_fan(
                mu=self.mu, m_i=m_i, lambda_t=self.electric_price[self.step_idx], len_time_slot=self.len_time_slot)
        # total_power_cost += power_cost
        # power_cost_rec[i] = power_cost

        # T_i_rec[:, i] = T_i
        # RAT_rec[i] = RAT_i

        observation = np.array([RAT_i, DAT_i, self.prev_DAS, self.prev_SPS])

        reward_part = {}
        reward_part['power_cost'] = power_cost / 10
        reward_part['max_DAS'] = - positive_part(DAS_i - self.max_DAS)

        reward_part['fluc_DAS'] = - \
            positive_part(np.abs(DAS_i - self.prev_DAS) - 1.5)
        reward_part['fluc_SPS'] = - \
            positive_part(np.abs(SPS_i - self.prev_SPS) - 0.3)

        # TODO add other reward here
        done = (self.step_idx == self.traj_len - 1)
        info = {'task': self.get_task(),
                'T_i': self.T_i,
                'DAS': DAS_i,
                'SPS': SPS_i,
                'power_cost': reward_part['power_cost'],
                'max_DAS': reward_part['max_DAS'],
                'fluc_DAS': reward_part['fluc_DAS'],
                'fluc_SPS': reward_part['fluc_SPS']}

        reward = sum(reward_part.values())

        self.prev_DAS, self.prev_SPS = DAS_i, SPS_i

        self.step_idx += 1

        return observation, reward, done, info

    def reset(self, options=None, seed=None):
        """
        Reset the environment. This should *NOT* automatically reset the task!
        Resetting the task is handled in the varibad wrapper (see wrappers.py).
        """

        if options != None:
            traj_context = options['traj_context'][self.id]
            assert self.traj_len == len(
                traj_context), "traj_context length must be equal to max_episode_steps"
            self.OAT = traj_context
        else:
            self.OAT = self.get_traj_context(self.traj_len)

        self.n_building = 10
        C_a = 1  # J/g/C
        # N_ij = 1 means building i and building j is connected
        N_ij = np.zeros((self.n_building, self.n_building))
        # thermal resistance between zones 𝑖 and 𝑗
        R_ij = np.zeros((self.n_building, self.n_building))
        for i in range(self.n_building):
            for j in range(i, self.n_building):
                N_ij[i, j] = np.random.randint(2)
                N_ij[j, i] = N_ij[i, j]

                R_ij[i, j] = np.random.normal(0.5, 0.1)
                R_ij[j, i] = R_ij[i, j]

        self.building_paras = {'n_building': self.n_building,
                               'C_i': np.random.uniform(10, 10, size=(self.n_building)),
                               'R_i': np.random.uniform(0.2, 0.2, size=(self.n_building)),
                               'N_ij': N_ij,
                               'R_ij': R_ij,
                               'C_a': 1}

        self.n_building = self.building_paras["n_building"]
        self.C_i = self.building_paras['C_i']
        self.R_i = self.building_paras['R_i']
        self.N_ij = self.building_paras['N_ij']
        self.R_ij = self.building_paras['R_ij']
        self.C_a = self.building_paras['C_a']

        self.a = self.AHU_paras['a']
        self.COP = self.AHU_paras['COP']
        self.mu = self.AHU_paras['mu']

        self.electric_price = np.random.uniform(
            0.2, 0.21, size=self.traj_len)
        # self.OAT = np.random.normal(95, 2, size=(self.traj_len))
        # self.OAT = np.sin(np.arange(self.traj_len) / 15) * \
        #     20 + 80 + np.random.normal(0, 1, size=(self.traj_len))

        self.step_idx = 0

        self.T_i_0 = np.random.uniform(90, 95, size=self.n_building)

        # 5_30 just put something here, not sure whether is the correct one
        # [RAT_i, DAT_i, self.prev_DAS, self.prev_SPS]
        return np.array([self.T_i_0.mean(), self.T_i_0.mean(), self.T_i_0.mean(), self.T_i_0.mean()])

    def get_task(self):
        """
        Return a task description, such as goal position or target velocity.
        """
        return self.AHU_paras, self.building_paras

    def reset_task(self, task=None):
        """
        Reset the task, either at random (if task=None) or the given task.
        Should *not* reset the environment.
        """
        pass

    def _get_T_t_plus_1(self, T_i, m_i, T_s, len_time_slot, OAT):
        """Get the temperature of each zone at time slot t+1.

        Refer to page 6 in ppt.

        Args:
            T_i(_type_): temperature of each zone at time slot t.
            m_i (_type_): air supply rate (in g/s) of zone 𝑖 at time slot t.
            T_s(_type_): supply air temperature of VFD
            len_time_slot(_type_): length of one time slot
            OAT: outside air temperature at time t

        Returns:
            T_i_plus_1: temperature of each zone at time slot t+1.
            RAT_plus_1: return air temperature at time slot t+1.
        """
        q_i = np.random.normal(
            0, 10, size=self.n_building)  # a random noise added to the final result

        a_i = np.zeros(self.n_building)
        for i in range(self.n_building):
            a_i[i] = 1 - len_time_slot / self.R_i[i] / self.C_i[i] - \
                np.sum(self.N_ij[i, :] * len_time_slot /
                       (self.R_ij[i, :] * self.C_i))

        b_ij = np.zeros((self.n_building, self.n_building))
        for i in range(self.n_building):
            for j in range(self.n_building):
                b_ij[i, j] = len_time_slot / (self.R_ij[i, j] * self.C_i[i])

        e_i = len_time_slot / (self.R_i * self.C_i)
        d_i = len_time_slot * self.C_a / self.C_i
        f_i = len_time_slot / self.C_i

        T_i_plus_1 = 0
        T_i_plus_1 += a_i * T_i

        for i in range(self.n_building):
            T_i_plus_1[i] += np.sum(self.N_ij[i, :] * b_ij[i, :] * T_i)

        T_i_plus_1 += d_i * m_i * (T_s - T_i)
        T_i_plus_1 += e_i * OAT + f_i * q_i
        RAT_plus_1 = np.sum(m_i * T_i_plus_1) / np.sum(m_i)

        return T_i_plus_1, RAT_plus_1

    def _get_cost_cooling_coil(self, m_i, T_s, MAT, COP, lambda_t, len_time_slot, eta):
        """Get energy cost related to cooling coil. Refer to page 8 in ppt.

        Args:
            m_i (_type_):  air supply rate (in g/s) of zone 𝑖 at time slot 𝑡
            T_s(_type_): supply air temperature of VFD, which is DAT
            MAT(_type_): mixed air temperature(outside air and return air)
            COP(_type_): oefficient of performance related the chiller
            lambda_t(_type_): electricity price at time t
            len_time_slot(_type_): time slot length
            eta(_type_): efficiency factor of the cooling coil
        Returns:
            p_t: power consumption cost of cooling coil
        """
        p_t = - positive_part(self.C_a * np.sum(m_i) * (MAT - T_s) / eta /
                              COP * lambda_t * len_time_slot)
        return p_t

    def _get_cost_supply_fan(self, mu, m_i, lambda_t, len_time_slot):
        """Get energy cost related to supply fan. Refer to page 9 in ppt.

        Args:
            mu(_type_): a coefficient
            m_i (_type_): air supply rate (in g/s) of zone 𝑖 at time slot 𝑡
            lambda_t(_type_): electricity price
            len_time_slot(_type_): time slot length

        Returns:
            _type_: _description_
        """
        return - positive_part(mu * np.sum(m_i ** 3) * lambda_t * len_time_slot)

    @ staticmethod
    def get_traj_context(traj_len):
        """
        Treat OAT as traj_context
        """
        OAT = np.sin(np.arange(traj_len) / 15) * \
            20 + 80 + np.random.normal(0, 1, size=(traj_len))
        return OAT

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

        unwrapped_env = env.venv.unwrapped.envs[0].unwrapped

        # --- initialise things we want to keep track of ---

        prev_obs = []
        next_obs = []
        actions = []
        rewards = []

        tasks = []
        T_i_rec = []
        DAS_rec = []
        SPS_rec = []
        power_cost_rec = []
        max_DAS_rec = []
        fluc_DAS_rec = []
        fluc_SPS_rec = []

        curr_direction = []
        forward_velocity = []

        if encoder is not None:

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

        if args.learner_type == 'secbad':
            hidden_rec = HiddenRecoder(encoder)
            p_G_t_dist_rec = []

        # curr_rollout_rew = []
        # pos.append(start_pos)

        # episode_tasks.append([])
        # T_i_rec.append([])

        if encoder is not None:
            # if episode_idx == 0:
            # reset to prior
            if args.learner_type == 'secbad':
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

        if args.learner_type == 'secbad':
            # G_t_dist = {1: 1}
            p_G_t_dist = {1: 1}
            best_unchange_length_rec = []

        iterator = progressbar.progressbar(
            range(1, env._max_episode_steps + 1), redirect_stdout=True) if args.learner_type == 'secbad' else range(1, env._max_episode_steps + 1)

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
            # T_i_rec[-1].append(info[0]['curr_target'])
            T_i_rec.append(info[0]['T_i'])
            DAS_rec.append(info[0]['DAS'])
            SPS_rec.append(info[0]['SPS'])
            power_cost_rec.append(info[0]['power_cost'])
            max_DAS_rec.append(info[0]['max_DAS'])
            fluc_DAS_rec.append(info[0]['fluc_DAS'])
            fluc_SPS_rec.append(info[0]['fluc_SPS'])

            # curr_direction.append(info[0]['curr_direction'])
            # forward_velocity.append(info[0]['forward_velocity'])

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

                elif args.learner_type == 'secbad':
                    # 5_23 use evaluate function in mixed_learner.py
                    state_decoder = kwargs['state_decoder']
                    reward_decoder = kwargs['reward_decoder']
                    # prev_state = episode_prev_obs[episode_idx][-1]
                    prev_state = prev_obs[-1]

                    inaccurate_priori = False if 'inaccurate_priori' not in kwargs else kwargs[
                        'inaccurate_priori']

                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, best_unchange_length, p_G_t_dist = MixedLearner.inference(
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
        if args.learner_type == 'secbad':
            p_G_t_dist_mat = np.zeros(
                (len(p_G_t_dist_rec)+1, len(p_G_t_dist_rec)+1))
            for i in range(len(p_G_t_dist_rec)):
                p_G_t_dist_mat[i, :i +
                               2] = list(p_G_t_dist_rec[i].values())[::-1]

        vis_dict = {'T_i_rec': T_i_rec,
                    'DAS': DAS_rec,
                    'SPS': SPS_rec,
                    'power_cost': power_cost_rec,
                    'max_DAS': max_DAS_rec,
                    'fluc_DAS': fluc_DAS_rec,
                    'fluc_SPS': fluc_SPS_rec,
                    'tasks': tasks,
                    'curr_direction': curr_direction,
                    'forward_velocity': forward_velocity,
                    'best_unchange_length_rec': best_unchange_length_rec if args.learner_type == 'secbad' else None,
                    'p_G_t_dist_rec': p_G_t_dist_rec if args.learner_type == 'secbad' else None,
                    'p_G_t_dist_mat': p_G_t_dist_mat if args.learner_type == 'secbad' else None,
                    }

        np.save('{}/{}_vis_data.npy'.format(image_folder, iter_idx), vis_dict)

        # plot the movement of ant
        # what to plot
        #
        # location x vs location y
        # task angle vs step, real angle vs step
        # vel along task angle vs step

        tasks = np.array(vis_dict['tasks'])
        T_i_rec = np.array(vis_dict['T_i_rec'])
        DAS_rec = np.array(vis_dict['DAS'])
        SPS_rec = np.array(vis_dict['SPS'])
        power_cost_rec = np.array(vis_dict['power_cost'])
        max_DAS_rec = np.array(vis_dict['max_DAS'])
        fluc_DAS_rec = np.array(vis_dict['fluc_DAS'])
        fluc_SPS_rec = np.array(vis_dict['fluc_SPS'])

        # TODO
        num_subplot = 7 + (args.learner_type == 'secbad')
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 3, 1)
        plt.plot(T_i_rec, c='r', alpha=0.3)
        plt.plot(unwrapped_env.OAT[:200], c='k', label='OAT')
        plt.hlines(unwrapped_env.max_DAS, xmin=0, xmax=200, label='max_DAS')
        plt.legend()
        plt.title('T_i')

        plt.subplot(3, 3, 2)
        plt.plot(DAS_rec)
        plt.title('DAS')

        plt.subplot(3, 3, 3)
        plt.plot(SPS_rec)
        plt.title('SPS')

        plt.subplot(3, 3, 4)
        plt.plot(power_cost_rec)
        plt.title('power_cost')

        plt.subplot(3, 3, 5)
        plt.plot(max_DAS_rec)
        plt.title('max_DAS')

        plt.subplot(3, 3, 6)
        plt.plot(fluc_DAS_rec)
        plt.title('fluc_DAS')

        plt.subplot(3, 3, 7)
        plt.plot(fluc_SPS_rec)
        plt.title('fluc_SPS')

        # plt.scatter(T_i_rec[:, 0], T_i_rec[:,
        #             1], c=np.arange(len(T_i_rec)))

        # info = {'task': self.get_task(),
        #         'T_i': self.T_i,
        #         'DAS': DAS_i,
        #         'SPS': SPS_i,
        #         'power_cost': reward_part['power_cost'],
        #         'max_DAS': reward_part['max_DAS'],
        #         'fluc_DAS': reward_part['fluc_DAS'],
        #         'fluc_SPS': reward_part['fluc_SPS']}

        # plt.colorbar()

        # plt.subplot(1, num_subplot, 2)
        # plt.plot(vis_dict['curr_direction'], range(
        #     len(vis_dict['curr_direction'])), 'k')
        # plt.plot(vis_dict['tasks'], range(len(vis_dict['tasks'])), 'r')
        # plt.xlim(0, 2*np.pi)

        # plt.subplot(1, num_subplot, 3)
        # plt.plot(vis_dict['forward_velocity'], range(
        #     len(vis_dict['curr_direction'])), 'k')
        # plt.title('velocity along task')

        if args.learner_type == 'secbad':
            plt.subplot(1, num_subplot, 4)
            plt.plot(vis_dict['best_unchange_length_rec'], range(
                len(vis_dict['best_unchange_length_rec'])), 'k')
            plt.xlabel('G_t', fontsize=15)

            # plt.subplot(1, num_subplot, 4)
            # p_G_t_dist

        plt.tight_layout()
        plt.savefig('{}/{}_behaviour'.format(image_folder, iter_idx))
        plt.close('all')

        if args.learner_type == 'secbad':
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


if __name__ == '__main__':
    hvac = HVAC()
    hvac.reset()

    #   info = {'task': self.get_task(),
    #         'T_i': self.T_i,
    #         'DAS': DAS_i,
    #         'SPS': SPS_i,
    #         'power_cost': reward_part['power_cost'],
    #         'max_DAS': reward_part['max_DAS'],
    #         'fluc_DAS': reward_part['fluc_DAS'],
    #         'fluc_SPS': reward_part['fluc_SPS']}

    T_i_rec = []
    DAS_rec = []
    SPS_rec = []
    power_cost_rec = []
    max_DAS_rec = []
    fluc_DAS_rec = []
    fluc_SPS_rec = []

    for i in range(200):
        action = np.random.normal(0, 1, 2)
        observation, reward, done, info = hvac.step(action)
        T_i_rec.append(info['T_i'])
        DAS_rec.append(info['DAS'])
        SPS_rec.append(info['SPS'])
        power_cost_rec.append(info['power_cost'])
        max_DAS_rec.append(info['max_DAS'])
        fluc_DAS_rec.append(info['fluc_DAS'])
        fluc_SPS_rec.append(info['fluc_SPS'])

    plt.figure(figsize=(10, 6))
    plt.subplot(3, 3, 1)
    plt.plot(T_i_rec, c='r', alpha=0.3)
    plt.plot(hvac.OAT[:200], c='k', label='OAT')
    plt.hlines(hvac.max_DAS, xmin=0, xmax=200, label='max_DAS')
    plt.legend()
    plt.title('T_i')

    plt.subplot(3, 3, 2)
    plt.plot(DAS_rec)
    plt.title('DAS')

    plt.subplot(3, 3, 3)
    plt.plot(SPS_rec)
    plt.title('SPS')

    plt.subplot(3, 3, 4)
    plt.plot(power_cost_rec)
    plt.title('power_cost')

    plt.subplot(3, 3, 5)
    plt.plot(max_DAS_rec)
    plt.title('max_DAS')

    plt.subplot(3, 3, 6)
    plt.plot(fluc_DAS_rec)
    plt.title('fluc_DAS')

    plt.subplot(3, 3, 7)
    plt.plot(fluc_SPS_rec)
    plt.title('fluc_SPS')

    plt.show()
