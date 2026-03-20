"""
synthetic neural dataset generation for model validation.
5 interpretable latent factors mixed into neurons with sparse weights and delays.
"""
import numpy as np


SYNTHETIC_DATA_PARAMS = dict(
    n_neurons=40,
    n_timesteps=100000,
    n_trials=500,
    noise_sd=0.1,
    p_participate=0.15,
    max_delay=10,
    tf_noise_sd=0.25,
    p_no_change=0.15,
    onset_tau=3.0,
    stim_tau=5.0,
    lick_tau=5.0,
    min_trial_length=3,
)


def _causal_exp_filter(x, tau=5):
    out = np.zeros_like(x)
    out[0] = x[0]
    a = 1.0 / tau
    for t in range(1, len(x)):
        out[t] = (1 - a) * out[t - 1] + a * x[t]
    return out


def generate_synthetic_data(
        n_neurons=40, n_timesteps=100000, n_trials=500,
        noise_sd=0.1, p_participate=0.15, max_delay=10,
        tf_noise_sd=0.25, p_no_change=0.15,
        change_magnitudes=(0.25, 0.43, 0.58, 1.0, 2.0),
        onset_tau=3.0, stim_tau=5.0, lick_tau=5.0,
        min_trial_length=10,
):
    """
    generate synthetic neural data driven by 5 latent factors:
        0  onset       exponential decay from trial start
        1  ramp        linear increase across trial
        2  stim_track  causally filtered stimulus
        3  pro_lick    fires when stimulus > 1, smoothed
        4  anti_lick   fires when stimulus < -1, smoothed

    returns dict with neural (nN, nT), factors (5, nT), stimulus, weights,
    delays, change_bins, trial_num, time_in_trial, factor_names
    """
    n_factors = 5
    factor_names = ['onset', 'ramp', 'stim_track', 'pro_lick', 'anti_lick']

    while True:
        breaks = np.sort(np.random.choice(
            np.arange(1, n_timesteps), n_trials - 1, replace=False))
        trial_lengths = np.diff(np.concatenate(([0], breaks, [n_timesteps])))
        if trial_lengths.min() >= min_trial_length:
            break

    participates = np.random.random((n_factors, n_neurons)) < p_participate
    weights = np.random.uniform(-1, 1, (n_factors, n_neurons)) * participates
    delays = np.random.randint(0, max_delay + 1, (n_factors, n_neurons)) * participates

    neural_all, factors_all, stim_all = [], [], []
    trial_num_all, time_in_trial_all = [], []
    change_bins = []

    for trial_idx, T in enumerate(trial_lengths):
        t = np.arange(T, dtype=float)

        stim = np.random.normal(0, tf_noise_sd, T)
        if np.random.random() > p_no_change:
            cb = np.random.randint(T // 2, max(T // 2 + 1, T - 1))
            stim[cb:] += np.random.choice(change_magnitudes)
            change_bins.append(cb)
        else:
            change_bins.append(None)

        f = np.zeros((T, n_factors))
        f[:, 0] = np.exp(-t / onset_tau)
        f[:, 1] = t / T
        f[:, 2] = _causal_exp_filter(stim, tau=stim_tau)
        f[:, 3] = _causal_exp_filter(np.maximum(stim - 1.0, 0), tau=lick_tau)
        f[:, 4] = _causal_exp_filter(np.maximum(-stim - 1.0, 0), tau=lick_tau)

        neural = np.random.normal(0, noise_sd, (T, n_neurons))
        for fac in range(n_factors):
            for neuron in range(n_neurons):
                if not participates[fac, neuron]:
                    continue
                d = int(delays[fac, neuron])
                if d < T:
                    neural[d:, neuron] += weights[fac, neuron] * f[:T - d, fac]

        neural_all.append(neural)
        factors_all.append(f)
        stim_all.append(stim)
        trial_num_all.append(np.full(T, trial_idx))
        time_in_trial_all.append(np.arange(T))

    neural = np.log1p(np.exp(np.concatenate(neural_all, axis=0))).T  # softplus, (nN, nT)
    factors = np.concatenate(factors_all, axis=0).T                   # (5, nT)
    stimulus = np.concatenate(stim_all)
    trial_num = np.concatenate(trial_num_all)
    time_in_trial = np.concatenate(time_in_trial_all)

    return dict(
        neural=neural, factors=factors, stimulus=stimulus,
        weights=weights, delays=delays, factor_names=factor_names,
        change_bins=change_bins,
        trial_num=trial_num, time_in_trial=time_in_trial,
    )
