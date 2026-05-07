"""
two-pulse interaction analysis: do mice integrate sequential fast TF pulses,
or respond to them independently?

replicates Fig 3c-d from Khilkevich & Lohse et al. 2024, split by block
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import ANALYSIS_OPTIONS, PLOT_OPTIONS
from behaviour.extraction import strip_and_convert_tf, binomial_ci


def extract_two_pulse_events(dfs, config=ANALYSIS_OPTIONS):
    """find fast TF pulse pairs at various delays during baseline,
    record whether a lick followed the second pulse"""
    frame_step = config.get('tf_sample_step', 3)
    frame_rate = config.get('frame_rate', 60)
    sample_rate = frame_rate / frame_step
    dt = 1 / sample_rate

    threshold = config.get('fast_pulse_threshold', 0.25)
    lick_win = config.get('tf_pulse_lick_win', [0.2, 1.5])
    max_delay = config.get('two_pulse_max_delay', 0.5)
    max_delay_samples = int(max_delay / dt)
    ignore_start = config.get('ignore_trial_start', 2)

    records = []
    for subj, df in dfs.items():
        for _, row in df.iterrows():
            tf_oct = strip_and_convert_tf(row['stim_TF'])
            tf_sub = tf_oct[::frame_step]

            change_time = row['stimT']
            lick_time = row['rt_FA'] if row['IsFA'] else np.inf
            cens_time = min(lick_time, change_time)

            start_sample = int(ignore_start * sample_rate)
            end_sample = min(int(cens_time * sample_rate), len(tf_sub))
            if end_sample <= start_sample:
                continue

            fast_idx = np.where(tf_sub[start_sample:end_sample] > threshold)[0]
            fast_idx += start_sample

            for j_pos, j in enumerate(fast_idx):
                pulse_time = j * dt
                licked = (lick_time >= pulse_time + lick_win[0] and
                          lick_time < pulse_time + lick_win[1])

                # find nearest preceding fast pulse
                nearest_delay = np.nan
                for i_pos in range(j_pos - 1, -1, -1):
                    gap = j - fast_idx[i_pos]
                    if gap > max_delay_samples:
                        break
                    nearest_delay = gap * dt

                records.append({
                    'subj': subj,
                    'block': row['hazardblock'],
                    'pulse_time': pulse_time,
                    'licked': licked,
                    'preceding_delay': nearest_delay,
                })

    return pd.DataFrame(records)


def calculate_two_pulse_interaction(dfs, config=ANALYSIS_OPTIONS):
    """compute interaction index J vs inter-pulse delay, per subject and block

    J = (P_pair - P_ind) / P_ind

    where P_ind = P_0 + 2*dP - dP^2 (inclusion-exclusion for two identical
    independent fast pulses) and dP = P_single - P_0
    """
    events = extract_two_pulse_events(dfs, config)

    delay_step = config.get('two_pulse_delay_step', 0.05)
    max_delay = config.get('two_pulse_max_delay', 0.5)
    delay_bins = np.arange(0, max_delay + delay_step, delay_step)
    delay_centres = delay_bins[:-1] + delay_step / 2

    lick_win = config.get('tf_pulse_lick_win', [0.2, 1.5])
    threshold = config.get('fast_pulse_threshold', 0.25)

    results = {}
    for subj in events['subj'].unique():
        results[subj] = {'delay_centres': delay_centres}

        for block in ['early', 'late']:
            subj_block = events[(events['subj'] == subj) &
                                (events['block'] == block)]

            # P_0: baseline lick rate from non-fast pulses in same block
            # computed from the full pulse data, not just the fast events
            # use the fast events to get P_single for isolated fast pulses
            isolated = subj_block[subj_block['preceding_delay'].isna()]
            paired = subj_block[subj_block['preceding_delay'].notna()]

            n_isolated = len(isolated)
            n_lick_isolated = isolated['licked'].sum()
            p_single = n_lick_isolated / n_isolated if n_isolated > 0 else np.nan

            # P_pair per delay bin
            p_pair = np.full(len(delay_centres), np.nan)
            p_pair_ci = np.full((len(delay_centres), 2), np.nan)
            n_pairs = np.zeros(len(delay_centres), dtype=int)

            for b in range(len(delay_centres)):
                in_bin = ((paired['preceding_delay'] >= delay_bins[b]) &
                          (paired['preceding_delay'] < delay_bins[b + 1]))
                n = in_bin.sum()
                n_pairs[b] = n
                if n > 0:
                    k = paired.loc[in_bin, 'licked'].sum()
                    p_pair[b] = k / n
                    p_pair_ci[b] = binomial_ci(k, n)

            results[subj][block] = {
                'p_single': p_single,
                'n_isolated': n_isolated,
                'p_pair': p_pair,
                'p_pair_ci': p_pair_ci,
                'n_pairs': n_pairs,
            }

    return results


def calculate_baseline_lick_rate(dfs, config=ANALYSIS_OPTIONS):
    """P_0: lick probability for TF samples near baseline (small deviations)"""
    from behaviour.extraction import extract_tf_pulses

    stim_df = extract_tf_pulses(dfs, config)
    lick_win = config.get('tf_pulse_lick_win', [0.2, 1.5])
    threshold = config.get('fast_pulse_threshold', 0.25)

    stim_df['licked'] = (
        stim_df['lick_time'].notna() &
        (stim_df['lick_time'] >= stim_df['stim_time'] + lick_win[0]) &
        (stim_df['lick_time'] < stim_df['stim_time'] + lick_win[1])
    )
    baseline_mask = np.abs(stim_df['tf_dev']) < threshold

    p0 = {}
    for subj in stim_df['subj'].unique():
        p0[subj] = {}
        for block in ['early', 'late']:
            mask = ((stim_df['subj'] == subj) &
                    (stim_df['block'] == block) & baseline_mask)
            sub = stim_df[mask]
            p0[subj][block] = sub['licked'].mean() if len(sub) > 0 else np.nan

    return p0


def compute_interaction_index(two_pulse_results, p0):
    """compute J = (P_pair - P_ind) / P_ind for each subject/block/delay"""
    results = {}
    for subj, subj_data in two_pulse_results.items():
        if subj == 'delay_centres':
            continue
        delay_centres = two_pulse_results[subj]['delay_centres']
        results[subj] = {'delay_centres': delay_centres}

        for block in ['early', 'late']:
            if block not in subj_data:
                continue
            data = subj_data[block]
            baseline = p0.get(subj, {}).get(block, np.nan)

            dp = data['p_single'] - baseline
            p_ind = baseline + 2 * dp - dp ** 2

            j = np.where(p_ind > 0,
                         (data['p_pair'] - p_ind) / p_ind, np.nan)

            results[subj][block] = {
                'J': j,
                'p_pair': data['p_pair'],
                'p_ind': p_ind,
                'p_single': data['p_single'],
                'p0': baseline,
                'n_pairs': data['n_pairs'],
            }

    return results


#%% plotting

def _block_colour(block, scale=1.0):
    r, g, b = [int(c * 255 * scale) for c in PLOT_OPTIONS['colours']['block'][block]]
    return f'rgb({r},{g},{b})'


def _block_rgba(block, alpha, scale=1.0):
    r, g, b = [int(c * 255 * scale) for c in PLOT_OPTIONS['colours']['block'][block]]
    return f'rgba({r},{g},{b},{alpha})'


def plot_two_pulse_interaction(interaction, min_n=20):
    """plot J vs inter-pulse delay, by block, with individual mice"""
    first_subj = next(s for s in interaction if 'delay_centres' in interaction[s])
    delay_centres = interaction[first_subj]['delay_centres']
    subjs = [s for s in interaction if s != first_subj and 'delay_centres' not in interaction[s]]
    subjs = [s for s in interaction if 'early' in interaction[s] or 'late' in interaction[s]]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)

    for block in ['early', 'late']:
        colour = _block_colour(block)
        colour_rgba = _block_rgba(block, 0.15)

        all_j = []
        for subj in subjs:
            if block not in interaction[subj]:
                continue
            data = interaction[subj][block]
            j = data['J'].copy()
            j[data['n_pairs'] < min_n] = np.nan
            all_j.append(j)

        if not all_j:
            continue
        all_j = np.stack(all_j)

        n_valid = np.sum(~np.isnan(all_j), axis=0).astype(float)
        mean_j = np.nanmean(all_j, axis=0)
        ci_j = 1.96 * np.nanstd(all_j, axis=0) / np.sqrt(
            np.where(n_valid > 0, n_valid, 1))

        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_j + ci_j,
            mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_j - ci_j,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=colour_rgba, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_j,
            mode='lines+markers', name=f'{block.capitalize()} block',
            line=dict(color=colour, width=2),
            marker=dict(size=5)), row=1, col=1)

    # delta: early - late
    all_early, all_late = [], []
    for subj in subjs:
        if 'early' not in interaction[subj] or 'late' not in interaction[subj]:
            continue
        je = interaction[subj]['early']['J'].copy()
        jl = interaction[subj]['late']['J'].copy()
        je[interaction[subj]['early']['n_pairs'] < min_n] = np.nan
        jl[interaction[subj]['late']['n_pairs'] < min_n] = np.nan
        all_early.append(je)
        all_late.append(jl)

    if all_early:
        deltas = np.stack(all_early) - np.stack(all_late)
        n_valid = np.sum(~np.isnan(deltas), axis=0).astype(float)
        mean_d = np.nanmean(deltas, axis=0)
        ci_d = 1.96 * np.nanstd(deltas, axis=0) / np.sqrt(
            np.where(n_valid > 0, n_valid, 1))

        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_d + ci_d,
            mode='lines', line=dict(width=0), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_d - ci_d,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(128,128,128,0.15)',
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_d,
            mode='lines+markers', name='Early - Late',
            line=dict(color='grey', width=2),
            marker=dict(size=5)), row=2, col=1)

    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'), row=1, col=1)
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'), row=2, col=1)

    fig.update_layout(template='plotly_white', width=500, height=500)
    fig.update_xaxes(title_text='Delay between fast pulses (s)', row=2, col=1)
    fig.update_yaxes(title_text='Interaction index (J)', row=1, col=1)
    fig.update_yaxes(title_text='Delta J (Early - Late)', row=2, col=1)

    return fig


def plot_two_pulse_raw(interaction, min_n=20):
    """plot observed P(lick|pair) vs independent prediction, by block"""
    first_subj = next(s for s in interaction if 'delay_centres' in interaction[s])
    delay_centres = interaction[first_subj]['delay_centres']
    subjs = [s for s in interaction if 'early' in interaction[s] or 'late' in interaction[s]]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=['Early block', 'Late block'])

    for col, block in enumerate(['early', 'late'], 1):
        colour = _block_colour(block)
        colour_rgba = _block_rgba(block, 0.15)

        all_pair = []
        all_ind = []
        for subj in subjs:
            if block not in interaction[subj]:
                continue
            data = interaction[subj][block]
            p = data['p_pair'].copy()
            p[data['n_pairs'] < min_n] = np.nan
            all_pair.append(p)
            all_ind.append(np.full_like(p, data['p_ind']))

        if not all_pair:
            continue
        all_pair = np.stack(all_pair)
        all_ind = np.stack(all_ind)

        n_valid = np.sum(~np.isnan(all_pair), axis=0).astype(float)
        mean_pair = np.nanmean(all_pair, axis=0)
        ci_pair = 1.96 * np.nanstd(all_pair, axis=0) / np.sqrt(
            np.where(n_valid > 0, n_valid, 1))
        mean_ind = np.nanmean(all_ind, axis=0)

        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_pair + ci_pair,
            mode='lines', line=dict(width=0), showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_pair - ci_pair,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=colour_rgba, showlegend=False), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_pair,
            mode='lines+markers', name='Observed',
            line=dict(color=colour, width=2), marker=dict(size=5),
            showlegend=(col == 1)), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=delay_centres, y=mean_ind,
            mode='lines', name='Independent prediction',
            line=dict(color='grey', width=2, dash='dash'),
            showlegend=(col == 1)), row=1, col=col)

    fig.update_layout(template='plotly_white', width=700, height=350)
    fig.update_xaxes(title_text='Delay between fast pulses (s)')
    fig.update_yaxes(title_text='P(lick | pair)', row=1, col=1)

    return fig
