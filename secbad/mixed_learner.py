from .environments.env_utils import VectorEnv, NonstationaryContext
import os
import time

from .utils.hidden_recoder import HiddenRecoder
# import non_envs

import gym
import numpy as np
import torch

# from .algorithms.a2c import A2C
from .algorithms.adaptive_online_storage import AdaptiveOnlineStorage
from .algorithms.ppo import PPO
# from environments.parallel_envs import make_vec_envs
from .models.policy import Policy
from .utils import evaluation as utl_eval
from .utils import helpers as utl
from .utils.tb_logger import TBLogger
from .vae import VaribadVAE

from scipy.stats import norm

import progressbar

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


# TODO put this function into a more appropriate place
# deprecat for HalfCheetahVelEnvNonstationary
# def get_r_t(curr_step_ls):
# # curr_step means the steps since the trajectory begin
# return torch.from_numpy(np.array([curr_step % 10 for curr_step in curr_step_ls], dtype=int)).to(
#     device).float().view((-1, 1))


class MixedLearner:
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
        self.iter_idx = 0

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label)

        # [ ] use SyncVectorEnv, AsyncVectorEnv will raise Pickle error. Fix it (or simply use the SyncVectorEnv instead?)
        # self.envs = gym.vector.AsyncVectorEnv([lambda: gym.make(
        #     id=args.env_name, traj_len=self.args.max_episode_steps) for _ in range(self.args.num_processes)])

        # context setter for both inference and training

        self.envs = VectorEnv(env_name=self.args.env_name,
                              n_env=self.args.num_processes, traj_len=self.args.traj_len, norm_rew=self.args.norm_rew_for_policy)

        vis_iter_ls = np.arange(0, self.num_updates, self.args.vis_interval)
        self.context_recoder = NonstationaryContext(
            self.envs.get_traj_context, vis_iter_ls, traj_len=self.args.traj_len)

        env_paras = self.envs.get_env_paras()

        self.args.action_dim = env_paras['dim_action']
        self.args.state_dim = env_paras['dim_state']
        self.args.context_dim = env_paras['dim_context']
        self.args.action_space = env_paras['action_space']

        # initialise VAE and policy
        self.vae = VaribadVAE(
            self.args, self.logger, lambda: self.iter_idx, env_paras['dim_context'])
        self.policy_storage = self.initialise_policy_storage()
        self.policy = self.initialise_policy()

    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()

        traj_context = [self.envs.get_traj_context(
            self.envs.traj_len) for i in range(self.envs.n_env)]
        prev_state = self.envs.reset(traj_context_ls=traj_context)

        # insert initial observation / embeddings to rollout storage
        self.policy_storage.prev_state[0].copy_(torch.tensor(prev_state))

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

            curr_step = 0
            for step in range(self.args.policy_num_steps):

                # sample actions from policy
                with torch.no_grad():
                    # value is the return predicted by the policy, action is the action suggested by the policy
                    value, action = utl.select_action(
                        args=self.args,
                        policy=self.policy,
                        state=torch.tensor(prev_state).float(),
                        # belief=belief,
                        # task=task,
                        deterministic=False,
                        latent_sample=latent_sample,
                        latent_mean=latent_mean,
                        latent_logvar=latent_logvar,
                    )

                # take step in the environment
                # [next_state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(
                #     self.envs, action, self.args)
                next_state, (rew_raw, rew_normalized), done, infos = self.envs.train_env_step(
                    action, self.args)

                # XXX infos 里面包含了 reward_forward & reward_control, 看起来 reward_forward 的数量级 没有比 reward_control 大太多
                # 是不是把 reward_forward 变大 （或者 reward_control 变小），否则出现的弊端之一是 agent 直接摆烂，让 reward_control = 0 最大化

                if self.args.learner_type == 'secbad':
                    r_t = torch.from_numpy(np.array([info['r_t'] for info in infos], dtype=int)).to(
                        device).float().view((-1, 1))  # r_t = 0 表示前一个循环 刚刚 reset 过 task

                # if r_t[0][0] == 0:
                # print('debug r_t: ', self.iter_idx, step)

                done = torch.from_numpy(np.array(done, dtype=int)).to(
                    device).float().view((-1, 1))
                # # create mask for episode ends
                # masks_done = torch.FloatTensor(
                #     [[0.0] if done_ else [1.0] for done_ in done]).to(device)
                # bad_mask is true if episode ended because time limit was reached
                # bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [
                #                               1.0] for info in infos]).to(device)

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
                        r_t=r_t if self.args.learner_type == 'secbad' else None)

                # before resetting, update the embedding and add to vae buffer
                # (last state might include useful task info)
                # XXX 这个 rollout_storage 会有影响么
                # ignore task decoder for now (task will be further treated as context decoder)
                task = None
                if not (self.args.disable_decoder and self.args.disable_kl_term):
                    self.vae.rollout_storage.insert(torch.tensor(prev_state),
                                                    action.detach().clone(),
                                                    torch.tensor(
                                                        next_state).clone(),
                                                    torch.tensor(
                                                        rew_raw).clone().unsqueeze(-1),
                                                    done.clone(),
                                                    task.clone() if task is not None else None,
                                                    r_ts=r_t.clone() if self.args.learner_type == 'secbad' else None)

                    self.vae.vae_buffer.insert(torch.tensor(prev_state),
                                               action.detach().clone(),
                                               torch.tensor(
                        next_state).clone(),
                        torch.tensor(
                        rew_raw).clone().unsqueeze(-1),
                        done.clone(),
                        step=curr_step,
                        r_ts=r_t.clone() if self.args.learner_type == 'secbad' else None)

                curr_step += 1
                # add the obs before reset to the policy storage
                self.policy_storage.next_state[step] = torch.tensor(next_state)

                # reset environments that are done
                # 6-17 Environments can reset automatically now
                # done_indices = np.argwhere(done.cpu().flatten()).flatten()
                # print('step, done_indices:', step, done_indices)
                # if len(done_indices) > 0:
                # next_state, belief, task, _ = utl.reset_env(self.envs, self.args,
                #                                             indices=done_indices, state=next_state)

                self.policy_storage.insert(
                    state=torch.tensor(next_state),
                    # belief=belief,
                    task=task,
                    actions=action,
                    rewards_raw=torch.tensor(rew_raw).unsqueeze(-1),
                    # 6-17 use rew_raw as rewards_normalised for now
                    rewards_normalised=torch.tensor(rew_raw).unsqueeze(-1),
                    value_preds=value,
                    # masks=masks_done,
                    # bad_masks=bad_masks,
                    done=done,
                    r_t=r_t if self.args.learner_type == 'secbad' else None,
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
                    train_stats = self.update(state=torch.tensor(prev_state).float(),
                                              #   belief=belief,
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

    def initialise_policy_storage(self):
        # AdaptiveOnlineStorage 这个东西应该记录的是 num_processes 这么多个 trajectory 从开始到结尾的 state latent belief task 这些
        return AdaptiveOnlineStorage(args=self.args,
                                     # XXX 'number of env steps to do (per process) before updating', 原来是 400, 是不是需要和原来的 max_traj_len * num_traj 相同？
                                     num_steps=self.args.policy_num_steps,
                                     num_processes=self.args.num_processes,
                                     state_dim=self.args.state_dim,
                                     latent_dim=self.args.latent_dim,
                                     #  belief_dim=self.args.belief_dim,
                                     context_dim=self.args.context_dim,
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
            # pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            dim_state=self.args.state_dim,
            dim_latent=self.args.latent_dim * 2,
            # dim_belief=self.args.belief_dim,
            dim_context=self.args.context_dim,
            #
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            #
            action_space=self.envs.get_env_paras()['action_space'],
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
        # elif self.args.policy == 'adaptive_ppo':  # PPO for adaptive learning and meta learning is the same
        #     policy = PPO(
        #         self.args,
        #         policy_net,
        #         self.args.policy_value_loss_coef,
        #         self.args.policy_entropy_coef,
        #         policy_optimiser=self.args.policy_optimiser,
        #         policy_anneal_lr=self.args.policy_anneal_lr,
        #         train_steps=self.num_updates,
        #         lr=self.args.lr_policy,
        #         eps=self.args.policy_eps,
        #         ppo_epoch=self.args.ppo_num_epochs,
        #         num_mini_batch=self.args.ppo_num_minibatch,
        #         use_huber_loss=self.args.ppo_use_huberloss,
        #         use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
        #         clip_param=self.args.ppo_clip_param,
        #         optimiser_vae=self.vae.optimiser_vae,
        #     )
        else:
            raise NotImplementedError

        return policy

    def visualize(self):

        visualize_index = self.args.visualize_index
        if visualize_index is None or (not os.path.exists(os.path.join(
                self.args.model_dir, 'policy{}.pt'.format(visualize_index)))):
            print('Incorrect visualize_index or None, use the lastest policy')
            visualize_index = max([int(f[len('policy'):-len('.pt')]) for f in os.listdir(self.args.model_dir) if
                                   (f.startswith('policy')) and len(f) > len('policy')+len('.pt')])

        if self.args.norm_rew_for_policy:
            ret_rms = utl.load_obj(
                self.args.model_dir, 'env_rew_rms{}'.format(visualize_index)) if self.args.norm_rew_for_policy else None

        self.policy.actor_critic.load_state_dict(torch.load(os.path.join(
            self.args.model_dir, 'policy{}.pt'.format(visualize_index)), map_location=device))

        if self.policy.actor_critic.pass_state_to_policy and self.policy.actor_critic.norm_state:
            self.policy.actor_critic.state_rms = utl.load_obj(
                self.args.model_dir, "env_state_rms{}".format(visualize_index))
        if self.policy.actor_critic.pass_latent_to_policy and self.policy.actor_critic.norm_latent:
            self.policy.actor_critic.latent_rms = utl.load_obj(
                self.args.model_dir, "env_latent_rms{}".format(visualize_index))
        # if self.policy.actor_critic.pass_belief_to_policy and self.policy.actor_critic.norm_belief:
        #     self.policy.actor_critic.belief_rms = utl.load_obj(
        #         self.args.model_dir, "env_belief_rms{}".format(visualize_index))
        if self.policy.actor_critic.pass_task_to_policy and self.policy.actor_critic.norm_task:
            self.policy.actor_critic.task_rms = utl.load_obj(
                self.args.model_dir, "env_task_rms{}".format(visualize_index))

        if self.args.decode_reward:
            self.vae.reward_decoder.load_state_dict(torch.load(os.path.join(
                self.args.model_dir, 'reward_decoder{}.pt'.format(visualize_index)), map_location=device))
        if self.args.decode_state:
            self.vae.state_decoder.load_state_dict(torch.load(os.path.join(
                self.args.model_dir, 'state_decoder{}.pt'.format(visualize_index)), map_location=device))
        if self.args.decode_task:
            self.vae.task_decoder.load_state_dict(torch.load(os.path.join(
                self.args.model_dir, 'task_decoder{}.pt'.format(visualize_index)), map_location=device))

        returns = utl_eval.visualise_behaviour(args=self.args,
                                               policy=self.policy,
                                               image_folder=self.logger.full_output_folder,
                                               iter_idx=visualize_index,
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
                                               rendering=True
                                               )

    def encode_running_trajectory(self):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :return:
        """

        # XXX 这里的prev_obs, next_obs, act, rew都是0，在truncate的时候是不是也要保证这些东西为0
        # for each process, get the current batch (zero-padded obs/act/rew + length indicators)
        prev_obs, next_obs, act, rew, lens = self.vae.rollout_storage.get_running_batch()
        prev_obs, next_obs, act, rew, lens = self.vae.vae_buffer.get_running_batch()

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

    # def get_value(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
    def get_value(self, state, latent_sample, latent_mean, latent_logvar):
        # get_latent_for_policy 这个函数没什么，只是按照 args 里 agent 能看到哪些东西 把 latent_sample, latent_mean, latent_logvar 拼起来
        latent = utl.get_latent_for_policy(
            self.args, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
        # 然后就是用 policy 的 V-network 算 value（似乎）
        # return self.policy.actor_critic.get_value(state=state, belief=belief, task=task, latent=latent).detach()
        return self.policy.actor_critic.get_value(state=state, latent=latent).detach()

    # def update(self, state, belief, task, latent_sample, latent_mean, latent_logvar):
    def update(self, state, task, latent_sample, latent_mean, latent_logvar):
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
                                            # belief=belief,
                                            # task=task,
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
        # 5_13 evaluate and visualize in the same frequency

        # --- visualise behaviour of policy ---

        if (self.iter_idx) % self.args.vis_interval == 0:
            ret_rms = self.envs.ret_rms if self.args.norm_rew_for_policy else None

            # returns = utl_eval.visualise_behaviour(args=self.args,
            #                                        policy=self.policy,
            #                                        image_folder=self.logger.full_output_folder,
            #                                        iter_idx=self.iter_idx,
            #                                        ret_rms=ret_rms,
            #                                        encoder=self.vae.encoder,
            #                                        reward_decoder=self.vae.reward_decoder,
            #                                        state_decoder=self.vae.state_decoder,
            #                                        task_decoder=self.vae.task_decoder,
            #                                        compute_rew_reconstruction_loss=self.vae.compute_rew_reconstruction_loss,
            #                                        compute_state_reconstruction_loss=self.vae.compute_state_reconstruction_loss,
            #                                        compute_task_reconstruction_loss=self.vae.compute_task_reconstruction_loss,
            #                                        compute_kl_loss=self.vae.compute_kl_loss,
            #                                     #    tasks=self.train_tasks,
            #                                        )

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

            # perform evaluation & inference and get the evaluation reward
            args = self.args

            task_dir_name = os.path.join(os.path.join(
                args.results_log_dir, 'logs_{}'.format(args.env_name), 'eval_traj_task_{}.npy'.format(args.seed)))

            # log the return avg/std across tasks (=processes)
            vis_context = self.context_recoder.load_context(
                task_dir_name, self.iter_idx, force_reset=True)

            prev_obs = []
            actions = []
            rewards = []

            info_rec = []

            if self.vae.encoder is not None:
                latent_samples = []
                latent_means = []
                latent_logvars = []
            else:
                curr_latent_sample = curr_latent_mean = curr_latent_logvar = None
                # episode_latent_samples = episode_latent_means = episode_latent_logvars = None
                latent_samples = latent_means = latent_logvars = None

            state = self.envs.reset_test(vis_context)
            if args.learner_type == 'secbad':
                hidden_rec = HiddenRecoder(self.vae.encoder)
                p_G_t_dist_rec = []

            if self.vae.encoder is not None:
                if args.learner_type == 'secbad':
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar = hidden_rec.encoder_init(
                        0)
                else:
                    curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = self.vae.encoder.prior(
                        1)
                    curr_latent_sample = curr_latent_sample[0].to(device)
                    curr_latent_mean = curr_latent_mean[0].to(device)
                    curr_latent_logvar = curr_latent_logvar[0].to(device)

                latent_samples.append(curr_latent_sample[0].clone())
                latent_means.append(curr_latent_mean[0].clone())
                latent_logvars.append(curr_latent_logvar[0].clone())

            if args.learner_type == 'secbad':
                # G_t_dist = {1: 1}
                p_G_t_dist = {1: 1}
                best_unchange_length_rec = []

            iterator = progressbar.progressbar(
                range(1, self.envs.traj_len + 1), redirect_stdout=True) if args.learner_type == 'secbad' else range(1, self.envs.traj_len + 1)

            for step_idx in iterator:
                prev_obs.append(state)

                latent = utl.get_latent_for_policy(args,
                                                   latent_sample=curr_latent_sample,
                                                   latent_mean=curr_latent_mean,
                                                   latent_logvar=curr_latent_logvar)
                #             _, action = policy.act(
                # state=state.view(-1), latent=latent, belief=belief, task=task, deterministic=True)
                _, action = self.policy.act(
                    state=torch.tensor(state).float(), latent=latent, task=None, deterministic=True)

                # (state, belief, task), (rew, rew_normalised), done, info = utl.env_step(
                #     env, action, args)

                state, (rew, rew_normalised), done, infos = self.envs.test_env_step(
                    action.numpy(), args)

                prev_obs.append(state)
                actions.append(action)
                rewards.append(rew)

                info_rec.append(infos)

                if self.vae.encoder is not None:
                    # update task embedding
                    if args.learner_type == 'varibad':
                        curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = self.vae.encoder(
                            action.reshape(
                                1, -1).float().to(device), torch.tensor(state).reshape(
                                1, -1).float().to(device), torch.tensor(rew).reshape(1, -1).float().to(device), hidden_state, return_prior=False)

                    elif args.learner_type == 'secbad':
                        # 5_23 use evaluate function in mixed_learner.py
                        state_decoder = self.vae.state_decoder
                        reward_decoder = self.vae.reward_decoder
                        # prev_state = episode_prev_obs[episode_idx][-1]
                        prev_state = prev_obs[-1]

                        curr_latent_sample, curr_latent_mean, curr_latent_logvar, best_unchange_length, p_G_t_dist = MixedLearner.inference(
                            hidden_rec, prev_state, action, state, rew, step_idx, reward_decoder, state_decoder, p_G=self.envs.get_p_G(args.inaccurate_priori), p_G_t_dist=p_G_t_dist)
                        best_unchange_length_rec.append(best_unchange_length)
                        p_G_t_dist_rec.append(p_G_t_dist)

                    elif args.learner_type == 'oracle_truncate':
                        # 5_23 use real reset point
                        raise NotImplemented

                    latent_samples.append(
                        curr_latent_sample[0].clone().to(device))
                    latent_means.append(curr_latent_mean[0].clone().to(device))
                    latent_logvars.append(
                        curr_latent_logvar[0].clone().to(device))

                rewards.append(rew)
                actions.append(action.reshape(1, -1).clone())

            episode_reward = np.sum(rewards)

            self.logger.add(
                'test_reward', episode_reward, self.iter_idx)

            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.frames}, "
                  f"FPS {int(self.frames / (time.time() - start_time))}, "
                  f"\n Return (evaluate): {episode_reward} \n"
                  )

            file_dir_name = os.path.join(
                self.logger.full_output_folder, 'behavior_{}.jpg'.format(self.iter_idx))
            self.envs.vis_traj(prev_obs, actions, rewards,
                               info_rec, file_dir_name)

        # --- save models ---

        if (self.iter_idx) % self.args.save_interval == 0:
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:
                # DONE save state_dict instead https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
                torch.save(self.policy.actor_critic.state_dict(), os.path.join(
                    save_path, f"policy{idx_label}.pt"))
                torch.save(self.vae.encoder.state_dict(), os.path.join(
                    save_path, f"encoder{idx_label}.pt"))
                if self.vae.state_decoder is not None:
                    torch.save(self.vae.state_decoder.state_dict(), os.path.join(
                        save_path, f"state_decoder{idx_label}.pt"))
                if self.vae.reward_decoder is not None:
                    torch.save(self.vae.reward_decoder.state_dict(), os.path.join(
                        save_path, f"reward_decoder{idx_label}.pt"))
                if self.vae.task_decoder is not None:
                    torch.save(self.vae.task_decoder.state_dict(), os.path.join(
                        save_path, f"task_decoder{idx_label}.pt"))

                # save normalisation params of envs
                # if self.args.norm_rew_for_policy:
                #     utl.save_obj(self.envs.venv.ret_rms, save_path,
                #                  f"env_rew_rms{idx_label}")
                if self.policy.actor_critic.pass_state_to_policy and self.policy.actor_critic.norm_state:
                    utl.save_obj(self.policy.actor_critic.state_rms,
                                 save_path, f"env_state_rms{idx_label}")
                if self.policy.actor_critic.pass_latent_to_policy and self.policy.actor_critic.norm_latent:
                    utl.save_obj(self.policy.actor_critic.latent_rms,
                                 save_path, f"env_latent_rms{idx_label}")
                # if self.policy.actor_critic.pass_belief_to_policy and self.policy.actor_critic.norm_belief:
                #     utl.save_obj(self.policy.actor_critic.belief_rms,
                #                  save_path, f"env_belief_rms{idx_label}")
                # if self.policy.actor_critic.pass_task_to_policy and self.policy.actor_critic.norm_task:
                #     utl.save_obj(self.policy.actor_critic.task_rms,
                #                  save_path, f"env_task_rms{idx_label}")
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

    def load_model(self):
        self.policy.actor_critic = torch.load(
            os.path.join(self.args.model_path), 'policy.pt')
        self.vae.encoder = torch.load(
            os.path.join(self.args.model_path), 'encoder.pt')
        if self.vae.state_decoder is not None:
            self.vae.state_decoder = torch.load(
                os.path.join(self.args.model_path), 'state_decoder.pt')
        if self.vae.reward_decoder is not None:
            self.vae.reward_decoder = torch.load(
                os.path.join(self.args.model_path), 'reward_decoder.pt')
        if self.vae.task_decoder is not None:
            self.vae.task_decoder = torch.load(
                os.path.join(self.args.model_path), 'task_decoder.pt')

    @ staticmethod
    def inference(hidden_rec, prev_state, action, state, rew, step_idx, reward_decoder, state_decoder, p_G, p_G_t_dist):
        # 5_23 compute the best reset point for all the non-stationary envs using secbad algorithm

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
            reward_mean = reward_decoder(
                latent_state=latent_samples, next_state=state, prev_state=prev_state, actions=action)
            if state_decoder is not None:
                state_mean = state_decoder(
                    latent_state=latent_samples, state=prev_state, actions=action)

            second_term = norm.pdf(
                rew.cpu().item(), loc=reward_mean.item(), scale=1)
            if state_decoder is not None:
                second_term *= np.prod(norm.pdf(state.squeeze(0).cpu(),
                                       loc=state_mean.squeeze(0).cpu(), scale=1))

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

        g_G_t_dist[1] = 0  # 5_24
        for k in range(1, step_idx+1):
            # g_G_t_dist[1] += p_G_t_dist[k] * \
            #     get_2nd_term_i_1(hidden_rec.get_record(
            #         reset_after=step_idx, up_to=step_idx, label='latent')) * p_G(G_t=1, G_t_minus_1=k)

            latent_samples, latent_mean, latent_logvar = hidden_rec.get_record(
                reset_after=step_idx, up_to=step_idx, label='latent')

            reward_mean = reward_decoder(
                latent_state=latent_samples, next_state=state, prev_state=prev_state, actions=action)
            if state_decoder is not None:
                state_mean = state_decoder(
                    latent_state=latent_samples, state=prev_state, actions=action)

            second_term = norm.pdf(
                rew.cpu().item(), loc=reward_mean.item(), scale=1)
            if state_decoder is not None:
                second_term *= np.prod(norm.pdf(state.squeeze(0).cpu(),
                                       loc=state_mean.squeeze(0).cpu(), scale=1))

            g_G_t_dist[1] += p_G_t_dist[k] * \
                second_term * p_G(G_t=1, G_t_minus_1=k)

        # get sum of g_G_t_dist
        sum_g_G_t = sum(g_G_t_dist.values())
        # divide each value of g_G_t_dist by sum_g_G_t
        # use for next iteration
        # print(sum_g_G_t)
        p_G_t_dist = {k: v / sum_g_G_t for k,
                      v in g_G_t_dist.items()}

        best_unchange_length = max(g_G_t_dist, key=g_G_t_dist.get)
        best_reset_after = step_idx + 1 - best_unchange_length

        # best_unchange_length_rec.append(best_unchange_length)

        curr_latent_sample, curr_latent_mean, curr_latent_logvar = hidden_rec.get_record(
            reset_after=best_reset_after, up_to=step_idx, label='latent')

        # print('reset_after: {}, up_to: {}'.format(best_reset_after, step_idx))

        assert curr_latent_sample.dim() == 2
        assert curr_latent_mean.dim() == 2
        assert curr_latent_logvar.dim() == 2

        return curr_latent_sample.clone(), curr_latent_mean.clone(), curr_latent_logvar.clone(), best_unchange_length, p_G_t_dist

        # episode_latent_samples[episode_idx].append(
        #     curr_latent_sample[0].clone())
        # episode_latent_means[episode_idx].append(
        #     curr_latent_mean[0].clone())
        # episode_latent_logvars[episode_idx].append(
        #     curr_latent_logvar[0].clone())
        # raise NotImplemented
