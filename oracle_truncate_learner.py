import os
import time

import gym
import numpy as np
import torch

from algorithms.a2c import A2C
from algorithms.adaptive_online_storage import AdaptiveOnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from vae import VaribadVAE


import progressbar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO put this function into a more appropriate place
# deprecat for HalfCheetahVelEnvNonstationary
# def get_r_t(curr_step_ls):
# # curr_step means the steps since the trajectory begin
# return torch.from_numpy(np.array([curr_step % 10 for curr_step in curr_step_ls], dtype=int)).to(
#     device).float().view((-1, 1))


# 其实这个文件的逻辑和 adaptive_learner.py 是完全一样的


class OracleTruncateLearner:
    """
    Meta-Learner class with the main training loop for variBAD.
    """

    def __init__(self, args):

        self.args = args
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = int(
            args.num_frames) // args.policy_num_steps // args.num_processes
        self.frames = 0
        self.iter_idx = -1

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        # initialise environments
        self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                  gamma=args.policy_gamma, device=device,
                                  episodes_per_task=self.args.max_rollouts_per_task,
                                  normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                  tasks=None
                                  )

        if self.args.single_task_mode:
            # get the current tasks (which will be num_process many different tasks)
            self.train_tasks = self.envs.get_task()
            # set the tasks to the first task (i.e. just a random task)
            self.train_tasks[1:] = self.train_tasks[0]
            # make it a list
            self.train_tasks = [t for t in self.train_tasks]
            # re-initialise environments with those tasks
            self.envs = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                                      gamma=args.policy_gamma, device=device,
                                      episodes_per_task=self.args.max_rollouts_per_task,
                                      normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                                      tasks=self.train_tasks
                                      )
            # save the training tasks so we can evaluate on the same envs later
            utl.save_obj(self.train_tasks,
                         self.logger.full_output_folder, "train_tasks")
        else:
            self.train_tasks = None

        # calculate what the maximum length of the trajectories is
        # num of steps per episode # default 200 for girdworld_nonstationary
        self.args.max_trajectory_len = self.envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.max_rollouts_per_task

        # get policy input dimensions
        self.args.state_dim = self.envs.observation_space.shape[0]
        self.args.task_dim = self.envs.task_dim
        self.args.belief_dim = self.envs.belief_dim
        self.args.num_states = self.envs.num_states
        # get policy output (action) dimensions
        self.args.action_space = self.envs.action_space
        if isinstance(self.envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = self.envs.action_space.shape[0]

        # initialise VAE and policy
        self.vae = VaribadVAE(self.args, self.logger, lambda: self.iter_idx)
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()

    def initialise_policy_storage(self):
        # AdaptiveOnlineStorage 这个东西应该记录的是 num_processes 这么多个 trajectory 从开始到结尾的 state latent belief task 这些
        return AdaptiveOnlineStorage(args=self.args,
                                     # XXX 'number of env steps to do (per process) before updating', 原来是 400, 是不是需要和原来的 max_traj_len * num_traj 相同？
                                     num_steps=self.args.policy_num_steps,
                                     num_processes=self.args.num_processes,
                                     state_dim=self.args.state_dim,
                                     latent_dim=self.args.latent_dim,
                                     belief_dim=self.args.belief_dim,
                                     task_dim=self.args.task_dim,
                                     action_space=self.args.action_space,
                                     hidden_size=self.args.encoder_gru_hidden_size,
                                     normalise_rewards=self.args.norm_rew_for_policy,
                                     detailed_reward=True
                                     )

    def initialise_policy(self):

        # initialise policy network
        policy_net = Policy(
            args=self.args,
            #
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_latent_to_policy=self.args.pass_latent_to_policy,
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim * 2,
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            #
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            #
            action_space=self.envs.action_space,
            init_std=self.args.policy_init_std,
        ).to(device)

        # initialise policy trainer
        if self.args.policy == 'a2c':
            policy = A2C(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                optimiser_vae=self.vae.optimiser_vae,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
            )
        elif self.args.policy == 'ppo':
            policy = PPO(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
                ppo_epoch=self.args.ppo_num_epochs,
                num_mini_batch=self.args.ppo_num_minibatch,
                use_huber_loss=self.args.ppo_use_huberloss,
                use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                clip_param=self.args.ppo_clip_param,
                optimiser_vae=self.vae.optimiser_vae,
            )
        elif self.args.policy == 'adaptive_ppo':  # PPO for adaptive learning and meta learning is the same
            policy = PPO(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
                ppo_epoch=self.args.ppo_num_epochs,
                num_mini_batch=self.args.ppo_num_minibatch,
                use_huber_loss=self.args.ppo_use_huberloss,
                use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                clip_param=self.args.ppo_clip_param,
                optimiser_vae=self.vae.optimiser_vae,
            )
        else:
            raise NotImplementedError

        return policy

    def train(self):
        # TODO truncate at right place
        """ Main Meta-Training loop """
        start_time = time.time()

        # reset environments
        prev_state, belief, task, info = utl.reset_env(self.envs, self.args)

        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(prev_state)

        # log once before training
        with torch.no_grad():
            self.log(None, None, start_time)

        # for self.iter_idx in range(self.num_updates):
        for self.iter_idx in progressbar.progressbar(range(self.num_updates), redirect_stdout=True):

            # First, re-compute the hidden states given the current rollouts (since the VAE might've changed)
            with torch.no_grad():
                latent_sample, latent_mean, latent_logvar, hidden_state = self.encode_running_trajectory()

            # add this initial hidden state to the policy storage
            # make sure we emptied buffers
            assert len(self.policy_storage.latent_mean) == 0
            self.policy_storage.hidden_states[0].copy_(hidden_state)
            self.policy_storage.latent_samples.append(latent_sample.clone())
            self.policy_storage.latent_mean.append(latent_mean.clone())
            self.policy_storage.latent_logvar.append(latent_logvar.clone())

            # rollout policies for a few steps
            # 和 env 交互，用得到的 trajectory 来算 encoding，并且把 policy_num_steps 这么多步的 encoding, action, 各种信息都存到 policy_storage 里面
            # XXX truncate 之后还需要用下面的 for loop 来算 encoding 么？
            # self.args.policy_num_steps 这个数 还是得大一点，起码在这么多次step内，env的task要变一次
            # 主要是这个循环得到的 encoding 后面 updating 的时候是怎么用的
            for step in range(self.args.policy_num_steps):

                # sample actions from policy
                with torch.no_grad():
                    # value is the return predicted by the policy, action is the action suggested by the policy
                    value, action = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=prev_state,
                        belief=belief,
                        task=task,
                        deterministic=False,
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )

                # take step in the environment
                [next_state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(
                    self.envs, action, self.args)

                # XXX infos 里面包含了 reward_forward & reward_control, 看起来 reward_forward 的数量级 没有比 reward_control 大太多
                # 是不是把 reward_forward 变大 （或者 reward_control 变小），否则出现的弊端之一是 agent 直接摆烂，让 reward_control = 0 最大化

                r_t = torch.from_numpy(np.array([info['r_t'] for info in infos], dtype=int)).to(
                    device).float().view((-1, 1))  # r_t = 0 表示前一个循环 刚刚 reset 过 task

                done = torch.from_numpy(np.array(done, dtype=int)).to(
                    device).float().view((-1, 1))
                # create mask for episode ends
                masks_done = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [
                                              1.0] for info in infos]).to(device)

                with torch.no_grad():
                    # compute next embedding (for next loop and/or value prediction bootstrap)
                    # 这里 是在 前一个 state 上，用 action, 得到 rew_raw 以及 next_state，对应的是 a_t-1, r_t, s_t, 和 paper 里 figure 2 对应
                    latent_sample, latent_mean, latent_logvar, hidden_state = utl.update_encoding(
                        encoder=self.vae.encoder,
                        next_obs=next_state,
                        action=action,
                        reward=rew_raw,
                        done=done,
                        hidden_state=hidden_state,
                        r_t=r_t)

                # before resetting, update the embedding and add to vae buffer
                # (last state might include useful task info)
                # XXX 这个 rollout_storage 会有影响么
                if not (self.args.disable_decoder and self.args.disable_kl_term):
                    self.vae.rollout_storage.insert(prev_state.clone(),
                                                    action.detach().clone(),
                                                    next_state.clone(),
                                                    rew_raw.clone(),
                                                    done.clone(),
                                                    task.clone() if task is not None else None,
                                                    r_ts=r_t.clone())

                # add the obs before reset to the policy storage
                self.policy_storage.next_state[step] = next_state.clone()

                # reset environments that are done
                done_indices = np.argwhere(done.cpu().flatten()).flatten()
                if len(done_indices) > 0:
                    next_state, belief, task, _ = utl.reset_env(self.envs, self.args,
                                                                indices=done_indices, state=next_state)

                # TODO: deal with resampling for posterior sampling algorithm
                #     latent_sample = latent_sample
                #     latent_sample[i] = latent_sample[i]

                # add experience to policy buffer # insert的是当前step结束后的状态
                if 'ctrl_reward' in info.keys():
                    self.policy_storage.insert(
                        state=next_state,
                        belief=belief,
                        task=task,
                        actions=action,
                        rewards_raw=rew_raw,
                        rewards_normalised=rew_normalised,
                        value_preds=value,
                        masks=masks_done,
                        bad_masks=bad_masks,
                        done=done,
                        r_t=r_t,
                        hidden_states=hidden_state.squeeze(0),
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                        ctrl_reward=torch.FloatTensor(
                            [info['ctrl_reward'] for info in infos]).unsqueeze(0).T.to(device),
                        vel_reward=torch.FloatTensor(
                            [info['vel_cost'] for info in infos]).unsqueeze(0).T.to(device)
                    )

                else:
                    self.policy_storage.insert(
                        state=next_state,
                        belief=belief,
                        task=task,
                        actions=action,
                        rewards_raw=rew_raw,
                        rewards_normalised=rew_normalised,
                        value_preds=value,
                        masks=masks_done,
                        bad_masks=bad_masks,
                        done=done,
                        r_t=r_t,
                        hidden_states=hidden_state.squeeze(0),
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )

                prev_state = next_state

                self.frames += self.args.num_processes

            # --- UPDATE ---
            # self.args.precollect_len 是为了在 self.vae.rollout_storage 里面记录 self.args.precollect_len 这么多个 trajectory，
            # 搜集了足够多的 traj 后，再进行 update
            if self.frames >= self.args.precollect_len:

                # check if we are pre-training the VAE
                if self.args.pretrain_len > self.iter_idx:
                    for p in range(self.args.num_vae_updates_per_pretrain):
                        self.vae.compute_vae_loss(update=True,
                                                  pretrain_index=self.iter_idx * self.args.num_vae_updates_per_pretrain + p)
                # otherwise do the normal update (policy + vae)
                else:
                    # 这里用的 prev_state, belief, task, latent_sample, latent_mean, latent_logvar 都是上一个 loop 的结果
                    # train_stats: value_loss_epoch, action_loss_epoch, dist_entropy_epoch, loss_epoch
                    train_stats = self.update(state=prev_state,
                                              belief=belief,
                                              task=task,
                                              latent_sample=latent_sample,
                                              latent_mean=latent_mean,
                                              latent_logvar=latent_logvar)

                    # log
                    run_stats = [
                        action, self.policy_storage.action_log_probs, value]
                    with torch.no_grad():
                        self.log(run_stats, train_stats, start_time)

            # clean up after update
            # XXX 这一步很关键，会把 policy_storage 的 latent_samples, latent_mean, latent_logvar 都清空
            # 但是 actions, prev_state, rewards_raw, hidden_states, value_preds 还留着，size 仍为 self.args.policy_num_steps，只不过如果 policy_storage 满了，在 insert 的时候是会从头覆盖原有的记录的
            self.policy_storage.after_update()

        self.envs.close()

    def encode_running_trajectory(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :return:
        """

        # XXX 这里的prev_obs, next_obs, act, rew都是0，在truncate的时候是不是也要保证这些东西为0
        # for each process, get the current batch (zero-padded obs/act/rew + length indicators)
        prev_obs, next_obs, act, rew, lens = self.vae.rollout_storage.get_running_batch()

        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        all_latent_samples, all_latent_means, all_latent_logvars, all_hidden_states = self.vae.encoder(actions=act,
                                                                                                       states=next_obs,
                                                                                                       rewards=rew,
                                                                                                       hidden_state=None,
                                                                                                       return_prior=True)

        # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
        latent_sample = (torch.stack(
            [all_latent_samples[lens[i]][i] for i in range(len(lens))])).to(device)
        latent_mean = (torch.stack([all_latent_means[lens[i]][i]
                       for i in range(len(lens))])).to(device)
        latent_logvar = (torch.stack(
            [all_latent_logvars[lens[i]][i] for i in range(len(lens))])).to(device)
        hidden_state = (torch.stack(
            [all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def get_value(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        # get_latent_for_policy 这个函数没什么，只是按照 args 里 agent 能看到哪些东西 把 latent_sample, latent_mean, latent_logvar 拼起来
        latent = utl.get_latent_for_policy(
            self.args, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
        # 然后就是用 policy 的 V-network 算 value（似乎）
        return self.policy.actor_critic.get_value(state=state, belief=belief, task=task, latent=latent).detach()

    def update(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:
        """
        # update policy (if we are not pre-training, have enough data in the vae buffer, and are not at iteration 0)
        if self.iter_idx >= self.args.pretrain_len and self.iter_idx > 0:

            # bootstrap next value prediction
            with torch.no_grad():  # 虽然是叫 actor_critic, 但如果用 PPO 的话，这个 actor_critic 实际上是用作 policy network & value function 的
                # XXX 算 当前 state, take given task, with network parametrized with given latent 得到的 reward
                next_value = self.get_value(state=state,
                                            belief=belief,
                                            task=task,
                                            latent_sample=latent_sample,
                                            latent_mean=latent_mean,
                                            latent_logvar=latent_logvar)

            # compute returns for current rollouts
            # XXX compute_returns 这个函数具体在算啥咱没搞懂，会把 self.policy_storage 的 returns 整个更新一遍
            # policy_gamma: discount factor for rewards
            # policy_tau: gae parameter
            # use_proper_time_limits: treat timeout and death differently (important in mujoco)
            self.policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                                self.args.policy_tau,
                                                use_proper_time_limits=self.args.use_proper_time_limits)

            # update agent (this will also call the VAE update!)
            policy_train_stats = self.policy.update(
                policy_storage=self.policy_storage,
                encoder=self.vae.encoder,
                rlloss_through_encoder=self.args.rlloss_through_encoder,
                compute_vae_loss=self.vae.compute_vae_loss)
        else:
            policy_train_stats = 0, 0, 0, 0

            # pre-train the VAE
            if self.iter_idx < self.args.pretrain_len:
                self.vae.compute_vae_loss(update=True)

        return policy_train_stats

    def log(self, run_stats, train_stats, start_time):

        # --- visualise behaviour of policy ---

        if (self.iter_idx + 1) % self.args.vis_interval == 0:
            ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            returns_per_episode = utl_eval.visualise_behaviour(args=self.args,
                                         policy=self.policy,
                                         image_folder=self.logger.full_output_folder,
                                         iter_idx=self.iter_idx,
                                         ret_rms=ret_rms,
                                         encoder=self.vae.encoder,
                                         reward_decoder=self.vae.reward_decoder,
                                         state_decoder=self.vae.state_decoder,
                                         task_decoder=self.vae.task_decoder,
                                         compute_rew_reconstruction_loss=self.vae.compute_rew_reconstruction_loss,
                                         compute_state_reconstruction_loss=self.vae.compute_state_reconstruction_loss,
                                         compute_task_reconstruction_loss=self.vae.compute_task_reconstruction_loss,
                                         compute_kl_loss=self.vae.compute_kl_loss,
                                         tasks=self.train_tasks,
                                         )

        # --- evaluate policy ----

        # if (self.iter_idx + 1) % self.args.eval_interval == 0:

            # ret_rms = self.envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            # returns_per_episode = utl_eval.evaluate(args=self.args,
            #                                         policy=self.policy,
            #                                         ret_rms=ret_rms,
            #                                         encoder=self.vae.encoder,
            #                                         iter_idx=self.iter_idx,
            #                                         tasks=self.train_tasks,
            #                                         )

            # log the return avg/std across tasks (=processes)

            returns_per_episode = torch.cat(returns_per_episode, dim=1)
            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            for k in range(len(returns_avg)):
                self.logger.add(
                    'return_avg_per_iter__episode_{}'.format(k + 1), returns_avg[k], self.iter_idx)
                self.logger.add(
                    'return_avg_per_frame__episode_{}'.format(k + 1), returns_avg[k], self.frames)
                self.logger.add(
                    'return_std_per_iter__episode_{}'.format(k + 1), returns_std[k], self.iter_idx)
                self.logger.add(
                    'return_std_per_frame__episode_{}'.format(k + 1), returns_std[k], self.frames)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - start_time))}, "
                  f"\n Mean return (evaluate): {returns_avg[-1].item()} \n"
                  )

        # --- save models ---

        if (self.iter_idx + 1) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:

                torch.save(self.policy.actor_critic, os.path.join(
                    save_path, f"policy{idx_label}.pt"))
                torch.save(self.vae.encoder, os.path.join(
                    save_path, f"encoder{idx_label}.pt"))
                if self.vae.state_decoder is not None:
                    torch.save(self.vae.state_decoder, os.path.join(
                        save_path, f"state_decoder{idx_label}.pt"))
                if self.vae.reward_decoder is not None:
                    torch.save(self.vae.reward_decoder, os.path.join(
                        save_path, f"reward_decoder{idx_label}.pt"))
                if self.vae.task_decoder is not None:
                    torch.save(self.vae.task_decoder, os.path.join(
                        save_path, f"task_decoder{idx_label}.pt"))

                # save normalisation params of envs
                if self.args.norm_rew_for_policy:
                    rew_rms = self.envs.venv.ret_rms
                    utl.save_obj(rew_rms, save_path, f"env_rew_rms{idx_label}")
                # TODO: grab from policy and save?
                # if self.args.norm_obs_for_policy:
                #     obs_rms = self.envs.venv.obs_rms
                #     utl.save_obj(obs_rms, save_path, f"env_obs_rms{idx_label}")

        # --- log some other things ---

        if ((self.iter_idx + 1) % self.args.log_interval == 0) and (train_stats is not None):

            self.logger.add('environment__state_max',
                            self.policy_storage.prev_state.max(), self.iter_idx)
            self.logger.add('environment__state_min',
                            self.policy_storage.prev_state.min(), self.iter_idx)

            self.logger.add('environment__rew_max',
                            self.policy_storage.rewards_raw.max(), self.iter_idx)
            self.logger.add('environment__rew_min',
                            self.policy_storage.rewards_raw.min(), self.iter_idx)
            self.logger.add('environment__rew_mean',
                            self.policy_storage.rewards_raw.mean(), self.iter_idx)
            self.logger.add('environment__rew_std',
                            self.policy_storage.rewards_raw.std(), self.iter_idx)
            self.logger.add('environment__rew_ctrl',
                            self.policy_storage.ctrl_reward.mean(), self.iter_idx)
            self.logger.add('environment__rew_vel',
                            self.policy_storage.vel_reward.mean(), self.iter_idx)

            self.logger.add('policy_losses__value_loss',
                            train_stats[0], self.iter_idx)
            self.logger.add('policy_losses__action_loss',
                            train_stats[1], self.iter_idx)
            self.logger.add('policy_losses__dist_entropy',
                            train_stats[2], self.iter_idx)
            self.logger.add('policy_losses__sum',
                            train_stats[3], self.iter_idx)

            self.logger.add('policy__action',
                            run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(self.policy.actor_critic, 'logstd'):
                self.logger.add(
                    'policy__action_logstd', self.policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add('policy__action_logprob',
                            run_stats[1].mean(), self.iter_idx)
            self.logger.add('policy__value',
                            run_stats[2].mean(), self.iter_idx)

            # self.logger.add(
            #     'encoder__latent_mean', torch.cat(self.policy_storage.latent_mean).mean(), self.iter_idx)
            # self.logger.add('encoder__latent_logvar',
            #                 torch.cat(self.policy_storage.latent_logvar).mean(), self.iter_idx, force_plot=True)

            self.logger.add('encoder_latent', torch.cat([torch.cat(self.policy_storage.latent_mean), torch.cat(
                self.policy_storage.latent_logvar)], dim=1).mean(), self.iter_idx)

            # log the average weights and gradients of all models (where applicable)
            for [model, name] in [
                [self.policy.actor_critic, 'policy'],
                [self.vae.encoder, 'encoder'],
                [self.vae.reward_decoder, 'reward_decoder'],
                [self.vae.state_decoder, 'state_transition_decoder'],
                [self.vae.task_decoder, 'task_decoder']
            ]:
                if model is not None:
                    param_list = list(model.parameters())
                    param_mean = np.mean(
                        [param_list[i].data.cpu().numpy().mean() for i in range(len(param_list))])
                    self.logger.add('weights__{}'.format(name),
                                    param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights__policy_std',
                                        param_list[0].data.mean(), self.iter_idx)
                    if param_list[0].grad is not None:
                        param_grad_mean = np.mean(
                            [param_list[i].grad.cpu().numpy().mean() for i in range(len(param_list))])
                        self.logger.add('gradients__{}'.format(name),
                                        param_grad_mean, self.iter_idx)
