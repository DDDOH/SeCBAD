import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class HiddenRecoder():
    # for evaluate a policy

    def __init__(self, encoder):
        self.encoder = encoder
        # dict of dict, rec[reset_after][up_to] = [curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state]
        self.rec = {}
        self.step = 0

        # 只要encoder参数不变，这个东西应该无论啥时候运行都是一样的，记录下来它的 return，节省计算资源
        self.encoder_prior = self.encoder.prior(1)
        self.encoder_prior = self.encoder_prior[0].squeeze(0), self.encoder_prior[1].squeeze(
            0), self.encoder_prior[2].squeeze(0), self.encoder_prior[3].squeeze(0),

    def _add_record(self, latent_sample, latent_mean, latent_logvar, hidden, reset_after, up_to):
        assert latent_sample.dim() == 2
        assert latent_mean.dim() == 2
        assert latent_logvar.dim() == 2
        if reset_after not in self.rec.keys():
            self.rec[reset_after] = {
                up_to: [latent_sample.clone().detach(),
                        latent_mean.clone().detach(),
                        latent_logvar.clone().detach(),
                        hidden.clone().detach()]}
        else:
            self.rec[reset_after][up_to] = [latent_sample.clone().detach(),
                                            latent_mean.clone().detach(),
                                            latent_logvar.clone().detach(),
                                            hidden.clone().detach()]

    def get_record(self, reset_after, up_to, label):
        # label can be 'hidden' or 'latent'
        assert reset_after is not None
        try:
            if label == 'hidden':
                return self.rec[reset_after][up_to][-1]
            elif label == 'latent':
                return self.rec[reset_after][up_to][0], self.rec[reset_after][up_to][1], self.rec[reset_after][up_to][2]
            else:
                raise ValueError('label can only be "hidden" or "latent"')
        except ValueError:
            raise ValueError("No record found")

    def encoder_init(self, step):
        curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = self.encoder_prior
        # reset after == current_step 的时候，这个 hidden_state 应该就是 encoder.prior
        self._add_record(curr_latent_sample, curr_latent_mean,
                         curr_latent_logvar, hidden_state, reset_after=step, up_to=step)
        return curr_latent_sample, curr_latent_mean, curr_latent_logvar

    def encoder_step(self, action, state, rew):
        # assume in this line self.step = 3, and the input action state rew come after step 4
        # best_reset_after takes value in [0,1,2,3,4]
        for i in range(self.step + 1):  # i in [0,1,2,3]
            # extend hidden state recording, for recordings reset after 0, 1, 2, 3
            prev_hidden_state = self.get_record(
                reset_after=i, up_to=self.step, label='hidden')
            curr_latent_sample, curr_latent_mean, curr_latent_logvar, hidden_state = self.encoder(action.reshape(1, -1).float().to(device), state, rew.reshape(1, -1).float().to(device),
                                                                                                  prev_hidden_state, return_prior=False)
            self._add_record(curr_latent_sample, curr_latent_mean,
                             curr_latent_logvar, hidden_state, reset_after=i, up_to=self.step+1)
            # if i == best_reset_after:
            #     return_latent = curr_latent_sample, curr_latent_mean, curr_latent_logvar

        # create hidden_state for reset after 4 and up to 4
        curr_latent_sample, curr_latent_mean, curr_latent_logvar = self.encoder_init(
            step=self.step + 1)
        # if best_reset_after == self.step + 1:
        #     return_latent = curr_latent_sample, curr_latent_mean, curr_latent_logvar

        self.step += 1

        # return return_latent

