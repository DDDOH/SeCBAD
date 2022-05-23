import warnings

import gym
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn

from models.decoder import StateTransitionDecoder, RewardDecoder, TaskDecoder
from models.encoder import RNNEncoder
from utils.helpers import get_task_dim, get_num_tasks
from utils.storage_vae import RolloutStorageVAE

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VaribadVAE:
    """
    VAE of VariBAD:
    - has an encoder and decoder
    - can compute the ELBO loss
    - can update the VAE (encoder+decoder)
    """

    def __init__(self, args, logger, get_iter_idx):

        self.args = args
        self.logger = logger
        self.get_iter_idx = get_iter_idx
        self.task_dim = get_task_dim(
            self.args) if self.args.decode_task else None
        self.num_tasks = get_num_tasks(
            self.args) if self.args.decode_task else None

        # initialise the encoder
        self.encoder = self.initialise_encoder()

        # initialise the decoders (returns None for unused decoders)
        self.state_decoder, self.reward_decoder, self.task_decoder = self.initialise_decoder()

        # initialise rollout storage for the VAE update
        # (this differs from the data that the on-policy RL algorithm uses)
        self.rollout_storage = RolloutStorageVAE(num_processes=self.args.num_processes,
                                                 max_trajectory_len=self.args.max_trajectory_len,
                                                 zero_pad=True,
                                                 max_num_rollouts=self.args.size_vae_buffer,
                                                 state_dim=self.args.state_dim,
                                                 action_dim=self.args.action_dim,
                                                 # probability of adding a new trajectory to buffer
                                                 vae_buffer_add_thresh=self.args.vae_buffer_add_thresh,
                                                 task_dim=self.task_dim
                                                 )

        # initalise optimiser for the encoder and decoders
        decoder_params = []
        if not self.args.disable_decoder:
            if self.args.decode_reward:
                decoder_params.extend(self.reward_decoder.parameters())
            if self.args.decode_state:
                decoder_params.extend(self.state_decoder.parameters())
            if self.args.decode_task:
                decoder_params.extend(self.task_decoder.parameters())
        self.optimiser_vae = torch.optim.Adam(
            [*self.encoder.parameters(), *decoder_params], lr=self.args.lr_vae)

    def initialise_encoder(self):
        """ Initialises and returns an RNN encoder """
        encoder = RNNEncoder(
            args=self.args,
            layers_before_gru=self.args.encoder_layers_before_gru,
            hidden_size=self.args.encoder_gru_hidden_size,
            layers_after_gru=self.args.encoder_layers_after_gru,
            latent_dim=self.args.latent_dim,
            action_dim=self.args.action_dim,
            action_embed_dim=self.args.action_embedding_size,
            state_dim=self.args.state_dim,
            state_embed_dim=self.args.state_embedding_size,
            reward_size=1,
            reward_embed_size=self.args.reward_embedding_size,
        ).to(device)
        return encoder

    def initialise_decoder(self):
        """ Initialises and returns the (state/reward/task) decoder as specified in self.args """

        if self.args.disable_decoder:
            return None, None, None

        latent_dim = self.args.latent_dim
        # if we don't sample embeddings for the decoder, we feed in mean & variance
        if self.args.disable_stochasticity_in_latent:
            latent_dim *= 2

        # initialise state decoder for VAE
        if self.args.decode_state:
            state_decoder = StateTransitionDecoder(
                args=self.args,
                layers=self.args.state_decoder_layers,
                latent_dim=latent_dim,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                pred_type=self.args.state_pred_type,
            ).to(device)
        else:
            state_decoder = None

        # initialise reward decoder for VAE
        if self.args.decode_reward:
            reward_decoder = RewardDecoder(
                args=self.args,
                layers=self.args.reward_decoder_layers,
                latent_dim=latent_dim,
                state_dim=self.args.state_dim,
                state_embed_dim=self.args.state_embedding_size,
                action_dim=self.args.action_dim,
                action_embed_dim=self.args.action_embedding_size,
                num_states=self.args.num_states,
                multi_head=self.args.multihead_for_reward,
                pred_type=self.args.rew_pred_type,
                input_prev_state=self.args.input_prev_state,
                input_action=self.args.input_action,
            ).to(device)
        else:
            reward_decoder = None

        # initialise task decoder for VAE
        if self.args.decode_task:
            assert self.task_dim != 0
            task_decoder = TaskDecoder(
                latent_dim=latent_dim,
                layers=self.args.task_decoder_layers,
                task_dim=self.task_dim,
                num_tasks=self.num_tasks,
                pred_type=self.args.task_pred_type,
            ).to(device)
        else:
            task_decoder = None

        return state_decoder, reward_decoder, task_decoder

    def compute_state_reconstruction_loss(self, latent, prev_obs, next_obs, action, return_predictions=False):
        """ Compute state reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

        state_pred = self.state_decoder(latent, prev_obs, action)

        if self.args.state_pred_type == 'deterministic':
            loss_state = (state_pred - next_obs).pow(2).mean(dim=-1)
        elif self.args.state_pred_type == 'gaussian':  # TODO: untested!
            state_pred_mean = state_pred[:, :state_pred.shape[1] // 2]
            state_pred_std = torch.exp(
                0.5 * state_pred[:, state_pred.shape[1] // 2:])
            m = torch.distributions.normal.Normal(
                state_pred_mean, state_pred_std)
            loss_state = -m.log_prob(next_obs).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_state, state_pred
        else:
            return loss_state

    def compute_rew_reconstruction_loss(self, latent, prev_obs, next_obs, action, reward, return_predictions=False):
        """ Compute reward reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

        if self.args.multihead_for_reward:
            rew_pred = self.reward_decoder(latent, None)
            if self.args.rew_pred_type == 'categorical':
                rew_pred = F.softmax(rew_pred, dim=-1)
            elif self.args.rew_pred_type == 'bernoulli':
                rew_pred = torch.sigmoid(rew_pred)

            env = gym.make(self.args.env_name)
            state_indices = env.task_to_id(next_obs).to(device)
            if state_indices.dim() < rew_pred.dim():
                state_indices = state_indices.unsqueeze(-1)
            rew_pred = rew_pred.gather(dim=-1, index=state_indices)
            rew_target = (reward == 1).float()
            if self.args.rew_pred_type == 'deterministic':  # TODO: untested!
                loss_rew = (rew_pred - reward).pow(2).mean(dim=-1)
            elif self.args.rew_pred_type in ['categorical', 'bernoulli']:
                loss_rew = F.binary_cross_entropy(
                    rew_pred, rew_target, reduction='none').mean(dim=-1)
            else:
                raise NotImplementedError
        else:
            rew_pred = self.reward_decoder(
                latent, next_obs, prev_obs, action.float())
            if self.args.rew_pred_type == 'bernoulli':  # TODO: untested!
                rew_pred = torch.sigmoid(rew_pred)
                rew_target = (reward == 1).float()  # TODO: necessary?
                loss_rew = F.binary_cross_entropy(
                    rew_pred, rew_target, reduction='none').mean(dim=-1)
            elif self.args.rew_pred_type == 'deterministic':
                loss_rew = (rew_pred - reward).pow(2).mean(dim=-1)
            else:
                raise NotImplementedError

        if return_predictions:
            return loss_rew, rew_pred
        else:
            return loss_rew

    def compute_task_reconstruction_loss(self, latent, task, return_predictions=False):
        """ Compute task reconstruction loss.
        (No reduction of loss along batch dimension is done here; sum/avg has to be done outside) """

        # 5_17 not sure how this part is used in training? Does it support non stationary task? fixing
        task_pred = self.task_decoder(latent)
        # task_pred of shape [traj_len, batch_size, 1]

        if self.args.task_pred_type == 'task_id':
            env = gym.make(self.args.env_name)
            task_target = env.task_to_id(task).to(device)
            # expand along first axis (number of ELBO terms)
            task_target = task_target.expand(task_pred.shape[:-1]).reshape(-1)
            loss_task = F.cross_entropy(task_pred.view(-1, task_pred.shape[-1]),
                                        task_target, reduction='none').view(task_pred.shape[:-1])
        elif self.args.task_pred_type == 'task_description':
            loss_task = (task_pred - task).pow(2).mean(dim=-1)
        else:
            raise NotImplementedError

        if return_predictions:
            return loss_task, task_pred
        else:
            return loss_task

    def compute_kl_loss(self, latent_mean, latent_logvar, elbo_indices):
        # -- KL divergence
        if self.args.kl_to_gauss_prior:
            kl_divergences = (- 0.5 * (1 + latent_logvar -
                              latent_mean.pow(2) - latent_logvar.exp()).sum(dim=-1))
        else:
            gauss_dim = latent_mean.shape[-1]
            # add the gaussian prior
            all_means = torch.cat(
                (torch.zeros(1, *latent_mean.shape[1:]).to(device), latent_mean))
            all_logvars = torch.cat(
                (torch.zeros(1, *latent_logvar.shape[1:]).to(device), latent_logvar))
            # https://arxiv.org/pdf/1811.09975.pdf
            # KL(N(mu,E)||N(m,S)) = 0.5 * (log(|S|/|E|) - K + tr(S^-1 E) + (m-mu)^T S^-1 (m-mu)))
            mu = all_means[1:]
            m = all_means[:-1]
            logE = all_logvars[1:]
            logS = all_logvars[:-1]
            kl_divergences = 0.5 * (torch.sum(logS, dim=-1) - torch.sum(logE, dim=-1) - gauss_dim + torch.sum(
                1 / torch.exp(logS) * torch.exp(logE), dim=-1) + ((m - mu) / torch.exp(logS) * (m - mu)).sum(dim=-1))

        # returns, for each ELBO_t term, one KL (so H+1 kl's)
        if elbo_indices is not None:
            batchsize = kl_divergences.shape[-1]
            task_indices = torch.arange(batchsize).repeat(
                self.args.vae_subsample_elbos)
            kl_divergences = kl_divergences[elbo_indices, task_indices].reshape(
                (self.args.vae_subsample_elbos, batchsize))

        return kl_divergences

    def compute_loss(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                     vae_rewards, vae_tasks, trajectory_lens, r_t=None, nonstationary_tasks=None):
        """
        Computes the VAE loss for the given data.
        Batches everything together and therefore needs all trajectories to be of the same length.
        (Important because we need to separate ELBOs and decoding terms so can't collapse those dimensions)
        """

        num_unique_trajectory_lens = len(np.unique(trajectory_lens))

        assert (num_unique_trajectory_lens == 1) or (
            self.args.vae_subsample_elbos and self.args.vae_subsample_decodes)
        assert not self.args.decode_only_past

        # cut down the batch to the longest trajectory length
        # this way we can preserve the structure
        # but we will waste some computation on zero-padded trajectories that are shorter than max_traj_len
        max_traj_len = np.max(trajectory_lens)
        # traj_len + 1 * batch_size * latent_dim
        latent_mean = latent_mean[:max_traj_len + 1]
        latent_logvar = latent_logvar[:max_traj_len + 1]
        # traj_len * batch_size * obs_dim/act_dim/rew_dim
        vae_prev_obs = vae_prev_obs[:max_traj_len]
        vae_next_obs = vae_next_obs[:max_traj_len]
        vae_actions = vae_actions[:max_traj_len]
        vae_rewards = vae_rewards[:max_traj_len]
        if nonstationary_tasks is not None:
            vae_nonstationary_tasks = nonstationary_tasks[:max_traj_len]

        # take one sample for each ELBO term
        if not self.args.disable_stochasticity_in_latent:
            latent_samples = self.encoder._sample_gaussian(
                latent_mean, latent_logvar)
        else:
            latent_samples = torch.cat((latent_mean, latent_logvar), dim=-1)

        # latent_samples: traj_len + 1 * batch_size * latent_dim
        num_elbos = latent_samples.shape[0]
        num_decodes = vae_prev_obs.shape[0]
        batchsize = latent_samples.shape[1]  # number of trajectories

        if r_t is None:
            # subsample elbo terms
            #   shape before: num_elbos * batchsize * dim
            #   shape after: vae_subsample_elbos * batchsize * dim
            # vae_subsample_elbos: for how many timesteps to compute the ELBO; None uses all'
            if self.args.vae_subsample_elbos is not None:
                # randomly choose which elbo's to subsample
                if num_unique_trajectory_lens == 1:
                    # select diff elbos for each task
                    elbo_indices = torch.LongTensor(
                        self.args.vae_subsample_elbos * batchsize).random_(0, num_elbos)
                else:
                    # XXX tfuck 这里在做什么， elbo_indices这个东西是干嘛的
                    # if we have different trajectory lengths, subsample elbo indices separately
                    # up to their maximum possible encoding length;
                    # only allow duplicates if the sample size would be larger than the number of samples
                    elbo_indices = np.concatenate([np.random.choice(range(0, t + 1), self.args.vae_subsample_elbos,
                                                                    replace=self.args.vae_subsample_elbos > (t+1)) for t in trajectory_lens])
                    if max_traj_len < self.args.vae_subsample_elbos:
                        warnings.warn('The required number of ELBOs is larger than the shortest trajectory, '
                                      'so there will be duplicates in your batch.'
                                      'To avoid this use --split_batches_by_elbo or --split_batches_by_task.')
                task_indices = torch.arange(batchsize).repeat(
                    self.args.vae_subsample_elbos)  # for selection mask

                latent_samples = latent_samples[elbo_indices, task_indices, :].reshape(
                    (self.args.vae_subsample_elbos, batchsize, -1))
                num_elbos = latent_samples.shape[0]
            else:
                elbo_indices = None

            # expand the state/rew/action inputs to the decoder (to match size of latents)
            # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
            dec_prev_obs = vae_prev_obs.unsqueeze(
                0).expand((num_elbos, *vae_prev_obs.shape))
            dec_next_obs = vae_next_obs.unsqueeze(
                0).expand((num_elbos, *vae_next_obs.shape))
            dec_actions = vae_actions.unsqueeze(
                0).expand((num_elbos, *vae_actions.shape))
            dec_rewards = vae_rewards.unsqueeze(
                0).expand((num_elbos, *vae_rewards.shape))

            # subsample reconstruction terms
            # vae_subsample_decodes: number of reconstruction terms to subsample; None uses all
            if self.args.vae_subsample_decodes is not None:
                # shape before: vae_subsample_elbos * num_decodes * batchsize * dim
                # shape after: vae_subsample_elbos * vae_subsample_decodes * batchsize * dim
                # (Note that this will always have duplicates given how we set up the code)

                # 5_15 如果segment不够长，允许重采样

                indices0 = torch.arange(num_elbos).repeat(
                    self.args.vae_subsample_decodes * batchsize)
                if num_unique_trajectory_lens == 1:
                    indices1 = torch.LongTensor(
                        num_elbos * self.args.vae_subsample_decodes * batchsize).random_(0, num_decodes)
                else:
                    indices1 = np.concatenate([np.random.choice(range(0, t), num_elbos * self.args.vae_subsample_decodes,
                                                                replace=True) for t in trajectory_lens])
                indices2 = torch.arange(batchsize).repeat(
                    num_elbos * self.args.vae_subsample_decodes)
                dec_prev_obs = dec_prev_obs[indices0, indices1, indices2, :].reshape(
                    (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
                dec_next_obs = dec_next_obs[indices0, indices1, indices2, :].reshape(
                    (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
                dec_actions = dec_actions[indices0, indices1, indices2, :].reshape(
                    (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
                dec_rewards = dec_rewards[indices0, indices1, indices2, :].reshape(
                    (num_elbos, self.args.vae_subsample_decodes, batchsize, -1))
                num_decodes = dec_prev_obs.shape[1]

            # expand the latent (to match the number of state/rew/action inputs to the decoder)
            # shape will be: [num tasks in batch] x [num elbos] x [len trajectory (reconstrution loss)] x [dimension]
            dec_embedding = latent_samples.unsqueeze(0).expand(
                (num_decodes, *latent_samples.shape)).transpose(1, 0)
        else:
            debug = False
            # 删掉 latent_samples 的最后一个 step，因为这个 step 其实是 RNN 最后的输出，并不会影响到任何一步的决策，如下图，latent_3并不影响任何东西
            # 但 priori latent 确实会影响第一步的决策
            """
                         priori latent      [latent_1, latent_2, latent_3]
                               ^                           ^
                               |                           |
            zero_hidden ---> [GRU] ---> hidden ---> [     GRU      ]
                                                           ^
                                                           |
                                            [s,a,r pair_1, pair_2, pair_3]
            
            """
            latent_samples = latent_samples[:-1, :, :]
            len_traj = latent_samples.shape[0]
            # to match the shape of latent_samples
            r_t = torch.cat([torch.zeros((1, *r_t.shape[1:])).to(device),
                            r_t], dim=0).squeeze(-1)[:-1, :]
            # segment_id: stores which segment each step belongs to, start from 0
            segment_id = torch.zeros(r_t.shape)
            for i in range(r_t.shape[1]):  # iterate over batch
                curr_segment = -1
                for j in range(r_t.shape[0]):  # iterate over step
                    if r_t[j, i] == 0:
                        curr_segment += 1
                    segment_id[j, i] = curr_segment

            batch_id = np.expand_dims(np.arange(batchsize), axis=0).repeat(
                len_traj, 0)

            # we have #batchsize trajectories of length len_traj
            # select self.args.vae_subsample_elbos time steps for each trajectory, use selected time steps as index for latent_samples
            # for each selected latent_samples index, select self.args.vae_subsample_decodes time steps, use selected time steps as index for dec_prev_obs, dec_next_obs, dec_actions, dec_rewards

            # output: dec_latent_samples of shape [vae_subsample_elbos, vae_subsample_decodes, batch_size, latent_dim]
            #         dec_latent_samples_segment_id of shape [vae_subsample_elbos, vae_subsample_decodes, batch_size]
            #         dec_latent_samples_batch_id of shape [vae_subsample_elbos, vae_subsample_decodes, batch_size]
            #         dec_prev_obs of shape [vae_subsample_elbos, vae_subsample_decodes, batch_size, obs_dim]
            #         dec_prev_obs_segment_id of shape [vae_subsample_elbos, vae_subsample_decodes, batch_size]
            #         dec_prev_obs_batch_id of shape [vae_subsample_elbos, vae_subsample_decodes, batch_size]
            # dec_latent_samples_segment_id and dec_prev_obs_segment_id should have the same value, ensure they come from the same segment
            # dec_latent_samples_batch_id and dec_prev_obs_batch_id should have the same value, ensure they come from the same trajectory

            # variables: latent_samples, vae_prev_obs
            # elbos_index [vae_subsample_elbos, batch_size] [i,k]: for k-th trajectory, the i-th (among self.args.vae_subsample_elbos) selected time step
            elbos_index = np.random.randint(0, len_traj, size=(
                self.args.vae_subsample_elbos, batchsize))
            if debug:
                # if debug, put 0 and len_traj-1 into elbos_index, make sure the first and last time step are selected into self.args.vae_subsample_elbos
                elbos_index[0, 0] = 0
                elbos_index[-1, -1] = len_traj-1

            # elbos_index[i,k] = elbos_index_flat[i * batchsize + k]
            elbos_index_flat = elbos_index.flatten()
            elbos_batch_index = np.expand_dims(np.arange(batchsize), axis=0).repeat(
                self.args.vae_subsample_elbos, 0)
            batch_index_flat = elbos_batch_index.flatten()

            # dec_latent_samples of shape [vae_subsample_elbos, vae_subsample_decodes, batch_size, latent_dim]
            # dec_latent_samples[i,j,k,:]: latent_sample for i-th elbo, j-th decode, k-th trajectory
            dec_embedding = latent_samples[elbos_index_flat,
                                           batch_index_flat, :]
            dec_embedding = dec_embedding.reshape(
                self.args.vae_subsample_elbos, batchsize, -1)
            # dec_latent_samples = np.expand_dims(dec_latent_samples, 1).repeat(
            #     self.args.vae_subsample_decodes, 1)
            dec_embedding = dec_embedding.unsqueeze(1).repeat_interleave(
                self.args.vae_subsample_decodes, 1)

            
            # dec_latent_samples_segment_id
            dec_latent_samples_segment_id = segment_id[elbos_index_flat,
                                                        batch_index_flat]
            dec_latent_samples_segment_id = dec_latent_samples_segment_id.reshape(
                self.args.vae_subsample_elbos, batchsize)
            dec_latent_samples_segment_id = np.expand_dims(dec_latent_samples_segment_id, 1).repeat(
                self.args.vae_subsample_decodes, 1)

            if debug:

                # dec_latent_samples_batch_id
                dec_latent_samples_batch_id = batch_id[elbos_index_flat,
                                                       batch_index_flat]
                dec_latent_samples_batch_id = dec_latent_samples_batch_id.reshape(
                    self.args.vae_subsample_elbos, batchsize)
                dec_latent_samples_batch_id = np.expand_dims(dec_latent_samples_batch_id, 1).repeat(
                    self.args.vae_subsample_decodes, 1)

            # decode_index [vae_subsample_elbos, vae_subsample_decodes, batch_size] [i, j, k]:
            # for k-th trajectory, the j-th (among self.args.vae_subsample_decodes) selected time step for i-th elbo time step

            # dec_latent_samples_segment_id[i,:,k]: segment id for i-th elbo, k-th trajectory, denote as segment_id_i_k
            # feasible_decode_index = np.where(segment_id[:,k] == segment_id_i_k)
            # decode_index[i,:,k] is np.random.choice(feasible_decode_index)

            # iterate over batchsize
            decode_index = np.zeros(
                (self.args.vae_subsample_elbos, self.args.vae_subsample_decodes, batchsize), dtype=np.int32)
            for k in range(batchsize):
                for i in range(self.args.vae_subsample_elbos):
                    segment_id_i_k = dec_latent_samples_segment_id[i, 0, k]
                    feasible_decode_index = np.where(
                        segment_id[:, k] == segment_id_i_k)[0]
                    # replace=True: a value can be selected multiple times
                    decode_index[i, :, k] = np.random.choice(
                        feasible_decode_index, self.args.vae_subsample_decodes, replace=True)
            if debug:
                for k in range(batchsize):
                    for i in range(self.args.vae_subsample_elbos):
                        segment_id_i_k = dec_latent_samples_segment_id[i, 0, k]
                        feasible_decode_index = np.where(
                            segment_id[:, k] == segment_id_i_k)[0]
                        # put the largest and smallest feasible_decode_index into decode_index
                        decode_index[i, 0, k] = max(feasible_decode_index)
                        decode_index[i, -1, k] = min(feasible_decode_index)

            # vae_prev_obs of shape [len_traj, batchsize, obs_dim]
            # dec_prev_obs of shape [vae_subsample_elbos, vae_subsample_decodes, batch_size, obs_dim]

            # elbos_index [vae_subsample_elbos, batch_size] [i,k]: for k-th trajectory, the i-th (among self.args.vae_subsample_elbos) selected time step
            # decode_index [vae_subsample_elbos, vae_subsample_decodes, batch_size] [i, j, k]:
            # for k-th trajectory, the j-th (among self.args.vae_subsample_decodes) selected time step for i-th elbo time step

            # 5_17 use for loop for now, can be vectorized to improve performance
            vae_nonstationary_task = nonstationary_tasks
            dec_prev_obs = torch.zeros(
                (self.args.vae_subsample_elbos, self.args.vae_subsample_decodes, batchsize, vae_prev_obs.shape[-1])).to(device)
            dec_actions = torch.zeros(
                (self.args.vae_subsample_elbos, self.args.vae_subsample_decodes, batchsize, vae_actions.shape[-1])).to(device)
            dec_rewards = torch.zeros(
                (self.args.vae_subsample_elbos, self.args.vae_subsample_decodes, batchsize, vae_rewards.shape[-1])).to(device)
            dec_next_obs = torch.zeros(
                (self.args.vae_subsample_elbos, self.args.vae_subsample_decodes, batchsize, vae_next_obs.shape[-1])).to(device)
            dec_nonstationary_task = torch.zeros(
                (self.args.vae_subsample_elbos, self.args.vae_subsample_decodes, batchsize, vae_nonstationary_task.shape[-1])).to(device)

            if debug:
                dec_prev_obs_segment_id = np.zeros(
                    (self.args.vae_subsample_elbos, self.args.vae_subsample_decodes, batchsize))
                dec_prev_obs_batch_id = np.zeros(
                    (self.args.vae_subsample_elbos, self.args.vae_subsample_decodes, batchsize))

            for i in range(self.args.vae_subsample_elbos):
                for j in range(self.args.vae_subsample_decodes):
                    for k in range(batchsize):
                        dec_prev_obs[i, j, k,
                                     :] = vae_prev_obs[decode_index[i, j, k], k, :]
                        dec_actions[i, j, k,
                                    :] = vae_actions[decode_index[i, j, k], k, :]
                        dec_rewards[i, j, k,
                                    :] = vae_rewards[decode_index[i, j, k], k, :]
                        dec_next_obs[i, j, k,
                                     :] = vae_next_obs[decode_index[i, j, k], k, :]
                        dec_nonstationary_task[i, j, k,
                                               :] = vae_nonstationary_task[decode_index[i, j, k], k, :]
            if debug:
                for i in range(self.args.vae_subsample_elbos):
                    for j in range(self.args.vae_subsample_decodes):
                        for k in range(batchsize):
                            dec_prev_obs_segment_id[i, j,
                                                    k] = segment_id[decode_index[i, j, k], k]
                            dec_prev_obs_batch_id[i, j,
                                                  k] = batch_id[decode_index[i, j, k], k]

                assert np.all(dec_latent_samples_segment_id ==
                              dec_prev_obs_segment_id)
                assert np.all(dec_latent_samples_batch_id ==
                              dec_prev_obs_batch_id)

                # dec_embedding = torch.tensor(
                #     dec_latent_samples).to(device).to(torch.float32)
                # dec_prev_obs = torch.tensor(dec_prev_obs).to(
                #     device).to(torch.float32)
                # dec_actions = torch.tensor(dec_actions).to(
                #     device).to(torch.float32)
                # dec_rewards = torch.tensor(dec_rewards).to(
                #     device).to(torch.float32)
                # dec_next_obs = torch.tensor(dec_next_obs).to(
                #     device).to(torch.float32)
                # dec_nonstationary_task = torch.tensor(
                #     dec_nonstationary_task).to(device).to(torch.float32)

        if self.args.decode_reward:
            # compute reconstruction loss for this trajectory (for each timestep that was encoded, decode everything and sum it up)
            # shape: [num_elbo_terms] x [num_reconstruction_terms] x [num_trajectories]

            rew_reconstruction_loss = self.compute_rew_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs,
                                                                           dec_actions, dec_rewards)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
            # avg/sum across individual reconstruction terms
            if self.args.vae_avg_reconstruction_terms:
                rew_reconstruction_loss = rew_reconstruction_loss.mean(dim=0)
            else:
                rew_reconstruction_loss = rew_reconstruction_loss.sum(dim=0)
            # average across tasks
            rew_reconstruction_loss = rew_reconstruction_loss.mean()
        else:
            rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = self.compute_state_reconstruction_loss(dec_embedding, dec_prev_obs,
                                                                               dec_next_obs, dec_actions)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                state_reconstruction_loss = state_reconstruction_loss.mean(
                    dim=0)
            else:
                state_reconstruction_loss = state_reconstruction_loss.sum(
                    dim=0)
            # avg/sum across individual reconstruction terms
            if self.args.vae_avg_reconstruction_terms:
                state_reconstruction_loss = state_reconstruction_loss.mean(
                    dim=0)
            else:
                state_reconstruction_loss = state_reconstruction_loss.sum(
                    dim=0)
            # average across tasks
            state_reconstruction_loss = state_reconstruction_loss.mean()
        else:
            state_reconstruction_loss = 0

        if self.args.decode_task:
            if r_t is not None:
                task_reconstruction_loss = self.compute_task_reconstruction_loss(
                    dec_embedding, dec_nonstationary_task)
            else:
                task_reconstruction_loss = self.compute_task_reconstruction_loss(
                    latent_samples, vae_tasks)
            # avg/sum across individual ELBO terms
            if self.args.vae_avg_elbo_terms:
                task_reconstruction_loss = task_reconstruction_loss.mean(dim=0)
            else:
                task_reconstruction_loss = task_reconstruction_loss.sum(dim=0)
            if r_t is not None:
                if self.args.vae_avg_reconstruction_terms:
                    task_reconstruction_loss = task_reconstruction_loss.mean(
                        dim=0)
                else:
                    task_reconstruction_loss = task_reconstruction_loss.sum(
                        dim=0)
                # average across tasks
                task_reconstruction_loss = task_reconstruction_loss.mean()
            else:
                # sum the elbos, average across tasks
                task_reconstruction_loss = task_reconstruction_loss.sum(
                    dim=0).mean()
        else:
            task_reconstruction_loss = 0

        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            # shape: [num_elbo_terms] x [num_trajectories]
            if r_t is not None:
                kl_loss = self.compute_kl_loss(
                    latent_mean, latent_logvar, elbos_index_flat)
            else:
                kl_loss = self.compute_kl_loss(
                    latent_mean, latent_logvar, elbo_indices)
            # avg/sum the elbos
            if self.args.vae_avg_elbo_terms:
                kl_loss = kl_loss.mean(dim=0)
            else:
                kl_loss = kl_loss.sum(dim=0)
            if r_t is not None:
                if self.args.vae_avg_reconstruction_terms:
                    kl_loss = kl_loss.mean(dim=0)
                else:
                    kl_loss = kl_loss.sum(dim=0)
                kl_loss = kl_loss.mean()
            else:
                # average across tasks
                kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss

    def compute_loss_split_batches_by_elbo(self, latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                                           vae_rewards, vae_tasks, trajectory_lens):
        """
        Loop over the elvo_t terms to compute losses per t.
        Saves some memory if batch sizes are very large,
        or if trajectory lengths are different, or if we decode only the past.
        """

        rew_reconstruction_loss = []
        state_reconstruction_loss = []
        task_reconstruction_loss = []

        assert len(np.unique(trajectory_lens)) == 1
        n_horizon = np.unique(trajectory_lens)[0]
        n_elbos = latent_mean.shape[0]  # includes the prior

        # for each elbo term (including one for the prior)...
        for idx_elbo in range(n_elbos):

            # get the embedding values (size: traj_length+1 * latent_dim; the +1 is for the prior)
            curr_means = latent_mean[idx_elbo]
            curr_logvars = latent_logvar[idx_elbo]

            # take one sample for each task
            if not self.args.disable_stochasticity_in_latent:
                curr_samples = self.encoder._sample_gaussian(
                    curr_means, curr_logvars)
            else:
                curr_samples = torch.cat((latent_mean, latent_logvar))

            # if the size of what we decode is always the same, we can speed up creating the batches
            if not self.args.decode_only_past:

                # expand the latent to match the (x, y) pairs of the decoder
                dec_embedding = curr_samples.unsqueeze(
                    0).expand((n_horizon, *curr_samples.shape))
                dec_embedding_task = curr_samples

                dec_prev_obs = vae_prev_obs
                dec_next_obs = vae_next_obs
                dec_actions = vae_actions
                dec_rewards = vae_rewards

            # otherwise, we unfortunately have to loop!
            # loop through the lengths we are feeding into the encoder for that trajectory (starting with prior)
            # (these are the different ELBO_t terms)
            else:

                # get the index until which we want to decode
                # (i.e. eithe runtil curr timestep or entire trajectory including future)
                if self.args.decode_only_past:
                    dec_from = 0
                    dec_until = idx_elbo
                else:
                    dec_from = 0
                    dec_until = n_horizon

                if dec_from == dec_until:
                    continue

                # (1) ... get the latent sample after feeding in some data (determined by len_encoder) & expand (to number of outputs)
                # num latent samples x embedding size
                dec_embedding = curr_samples.unsqueeze(0).expand(
                    dec_until - dec_from, *curr_samples.shape)
                dec_embedding_task = curr_samples
                # (2) ... get the predictions for the trajectory until the timestep we're interested in
                dec_prev_obs = vae_prev_obs[dec_from:dec_until]
                dec_next_obs = vae_next_obs[dec_from:dec_until]
                dec_actions = vae_actions[dec_from:dec_until]
                dec_rewards = vae_rewards[dec_from:dec_until]

            if self.args.decode_reward:
                # compute reconstruction loss for this trajectory (for each timestep that was encoded, decode everything and sum it up)
                # size: if all trajectories are of same length [num_elbo_terms x num_reconstruction_terms], otherwise it's flattened into one
                rrc = self.compute_rew_reconstruction_loss(dec_embedding, dec_prev_obs, dec_next_obs, dec_actions,
                                                           dec_rewards)
                # sum up the reconstruction terms; average over tasks
                rrc = rrc.sum(dim=0).mean()
                rew_reconstruction_loss.append(rrc)

            if self.args.decode_state:
                src = self.compute_state_reconstruction_loss(
                    dec_embedding, dec_prev_obs, dec_next_obs, dec_actions)
                # sum up the reconstruction terms; average over tasks
                src = src.sum(dim=0).mean()
                state_reconstruction_loss.append(src)

            if self.args.decode_task:
                trc = self.compute_task_reconstruction_loss(
                    dec_embedding_task, vae_tasks)
                # average across tasks
                trc = trc.mean()
                task_reconstruction_loss.append(trc)

        # sum the ELBO_t terms
        if self.args.decode_reward:
            rew_reconstruction_loss = torch.stack(rew_reconstruction_loss)
            rew_reconstruction_loss = rew_reconstruction_loss.sum()
        else:
            rew_reconstruction_loss = 0

        if self.args.decode_state:
            state_reconstruction_loss = torch.stack(state_reconstruction_loss)
            state_reconstruction_loss = state_reconstruction_loss.sum()
        else:
            state_reconstruction_loss = 0

        if self.args.decode_task:
            task_reconstruction_loss = torch.stack(task_reconstruction_loss)
            task_reconstruction_loss = task_reconstruction_loss.sum()
        else:
            task_reconstruction_loss = 0

        if not self.args.disable_kl_term:
            # compute the KL term for each ELBO term of the current trajectory
            kl_loss = self.compute_kl_loss(latent_mean, latent_logvar, None)
            # sum the elbos, average across tasks
            kl_loss = kl_loss.sum(dim=0).mean()
        else:
            kl_loss = 0

        return rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss

    def compute_vae_loss(self, update=False, pretrain_index=None):
        """ Returns the VAE loss """

        if not self.rollout_storage.ready_for_update():
            return 0

        if self.args.disable_decoder and self.args.disable_kl_term:
            return 0

        # get a mini-batch
        vae_prev_obs, vae_next_obs, vae_actions, vae_rewards, vae_tasks, \
            trajectory_lens, r_t, nonstationary_tasks = self.rollout_storage.get_batch(
                batchsize=self.args.vae_batch_num_trajs)
        # vae_prev_obs will be of size: max trajectory len x num trajectories x dimension of observations

        # pass through encoder (outputs will be: (max_traj_len+1) x number of rollouts x latent_dim -- includes the prior!)
        # FIXED 5_16 注意这里并没有引入何时 reset RNN hidden state 的概念，可能需要改
        _, latent_mean, latent_logvar, _ = self.encoder(actions=vae_actions,
                                                        states=vae_next_obs,
                                                        rewards=vae_rewards,
                                                        hidden_state=None,
                                                        return_prior=True,
                                                        detach_every=self.args.tbptt_stepsize if hasattr(
                                                            self.args, 'tbptt_stepsize') else None,  # tbptt_stepsize: stepsize for truncated backpropagation through time; None uses max (horizon of BAMDP)
                                                        r_t=r_t
                                                        )

        if self.args.split_batches_by_task:
            raise NotImplementedError
            losses = self.compute_loss_split_batches_by_task(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs,
                                                             vae_actions, vae_rewards, vae_tasks,
                                                             trajectory_lens, len_encoder)
        elif self.args.split_batches_by_elbo:
            losses = self.compute_loss_split_batches_by_elbo(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs,
                                                             vae_actions, vae_rewards, vae_tasks,
                                                             trajectory_lens)
        else:
            # FIXED 5_15 dimension of vae_tasks? does it change within one episode?
            losses = self.compute_loss(latent_mean, latent_logvar, vae_prev_obs, vae_next_obs, vae_actions,
                                       vae_rewards, vae_tasks, trajectory_lens, r_t=r_t, nonstationary_tasks=nonstationary_tasks)
        rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss = losses

        # VAE loss = KL loss + reward reconstruction + state transition reconstruction
        # take average (this is the expectation over p(M))
        loss = (self.args.rew_loss_coeff * rew_reconstruction_loss +
                self.args.state_loss_coeff * state_reconstruction_loss +
                self.args.task_loss_coeff * task_reconstruction_loss +
                self.args.kl_weight * kl_loss).mean()

        # make sure we can compute gradients
        if not self.args.disable_kl_term:
            assert kl_loss.requires_grad
        if self.args.decode_reward:
            assert rew_reconstruction_loss.requires_grad
        if self.args.decode_state:
            assert state_reconstruction_loss.requires_grad
        if self.args.decode_task:
            assert task_reconstruction_loss.requires_grad

        # overall loss
        elbo_loss = loss.mean()

        if update:
            self.optimiser_vae.zero_grad()
            elbo_loss.backward()
            # clip gradients
            if self.args.encoder_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    self.encoder.parameters(), self.args.encoder_max_grad_norm)
            if self.args.decoder_max_grad_norm is not None:
                if self.args.decode_reward:
                    nn.utils.clip_grad_norm_(
                        self.reward_decoder.parameters(), self.args.decoder_max_grad_norm)
                if self.args.decode_state:
                    nn.utils.clip_grad_norm_(
                        self.state_decoder.parameters(), self.args.decoder_max_grad_norm)
                if self.args.decode_task:
                    nn.utils.clip_grad_norm_(
                        self.task_decoder.parameters(), self.args.decoder_max_grad_norm)
            # update
            self.optimiser_vae.step()

        self.log(elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss,
                 pretrain_index)

        return elbo_loss

    def log(self, elbo_loss, rew_reconstruction_loss, state_reconstruction_loss, task_reconstruction_loss, kl_loss,
            pretrain_index=None):

        if pretrain_index is None:
            curr_iter_idx = self.get_iter_idx()
        else:
            curr_iter_idx = - self.args.pretrain_len * \
                self.args.num_vae_updates_per_pretrain + pretrain_index

        if curr_iter_idx % self.args.log_interval == 0:

            if self.args.decode_reward:
                self.logger.add('vae_losses__reward_reconstr_err',
                                rew_reconstruction_loss.mean(), curr_iter_idx)
            if self.args.decode_state:
                self.logger.add('vae_losses__state_reconstr_err',
                                state_reconstruction_loss.mean(), curr_iter_idx)
            if self.args.decode_task:
                self.logger.add('vae_losses__task_reconstr_err',
                                task_reconstruction_loss.mean(), curr_iter_idx)

            if not self.args.disable_kl_term:
                self.logger.add('vae_losses__kl',
                                kl_loss.mean(), curr_iter_idx)
            self.logger.add('vae_losses__sum', elbo_loss, curr_iter_idx)
