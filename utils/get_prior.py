import numpy as np


def get_prior(segment_len_mean, segment_len_std, segment_len_min, max_episode_steps, samples=10000):
    '''
    segment_len_mean, segment_len_std: Gaussian distribution segment_len_mean & segment_len_std 
    segment_len_min: max_transform (e.g., 10)
    max_episode_steps
    samples: how many samples to use

    return: p_G  # (max_episode_steps, max_episode_steps)
    # p_G[i, k]: p(G_t = i | G_{t-1} = k)
    '''

    length = max_episode_steps * 3
    q = np.zeros((samples, length), dtype=np.int32)

    for sample in range(samples):
        cum_t = 0
        while cum_t < length // 2:
            gau_random = max(
                round(np.random.normal(loc=segment_len_mean, scale=segment_len_std)), segment_len_min)
            q[sample, cum_t] = 1
            for i in range(1, gau_random):
                q[sample, cum_t + i] = q[sample, cum_t + i - 1] + 1
            cum_t += gau_random

    # from IPython import embed; embed()
    p_G = np.zeros((length, length))
    for i in range(samples):
        for j in range(1, max_episode_steps):
            p_G_t = q[i, j]  # p(G_t)
            p_G_t_1 = q[i, j-1]  # p(G_{t-1})
            p_G[p_G_t, p_G_t_1] += 1

    length = max_episode_steps

    p_G = p_G[:length+5, :length+5]

    for i in range(length+5):
        p_G[i] = p_G[i] / (np.sum(p_G[i]) + 1e-7)

    return p_G


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    pG = get_prior(20, 30, 3, 100, 1000)
    print(pG[10, 9])

    print(pG[1, 3])
    plt.imshow(pG)
    plt.colorbar()
    plt.show()

    # def p_G(G_t, G_t_minus_1):
    # # 5_17  update this function
    # if G_t - G_t_minus_1 == 1:
    #     return 1 - 1/unwrapped_env.change_interval_base
    # else:
    #     # G_t = 1, G_t_minus_1 = k
    #     return 1/unwrapped_env.change_interval_base
