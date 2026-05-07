"""
plotly-based visualisation for behavioural analyses
"""
import warnings
import numpy as np
import plotly.graph_objects as go

warnings.filterwarnings('ignore', message='Mean of empty slice')
warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d, gaussian_filter

from config import ANALYSIS_OPTIONS, PLOT_OPTIONS


def _rgb(colour_tuple, scale=1.0):
    """convert (0-1 float) colour tuple to plotly rgb string, with optional darkening"""
    r, g, b = [int(c * 255 * scale) for c in colour_tuple]
    return f'rgb({r},{g},{b})'


def _rgba(colour_tuple, alpha, scale=1.0):
    r, g, b = [int(c * 255 * scale) for c in colour_tuple]
    return f'rgba({r},{g},{b},{alpha})'


def _block_colour(block, scale=1.0):
    return _rgb(PLOT_OPTIONS['colours']['block'][block], scale)


def _block_rgba(block, alpha, scale=1.0):
    return _rgba(PLOT_OPTIONS['colours']['block'][block], alpha, scale)


def plot_psychometric(psycho_or_chrono, config=ANALYSIS_OPTIONS):
    change_tfs = config['change_tfs']

    if np.nanmax(psycho_or_chrono.flatten()) > 1:
        yax = 'RT (s)'
        delta_yax = 'delta RT (s)'
    else:
        yax = 'P(Hit)'
        delta_yax = 'delta P(Hit)'

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)

    for block_id, block in enumerate(['early', 'late']):
        for probe_id, probe in enumerate([False, True]):
            scale = 0.65 if probe else 1.0
            colour = _block_colour(block, scale)
            dash = 'dash' if probe else 'solid'
            label = f"{block} {'probe' if probe else ''}"

            data = psycho_or_chrono[:, :, block_id, probe_id]
            mean = np.nanmean(data, axis=0)
            sem = np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))

            fig.add_trace(go.Scatter(
                x=change_tfs, y=mean,
                mode='lines+markers', name=label,
                line=dict(color=colour, dash=dash)), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=list(change_tfs) + list(change_tfs)[::-1],
                y=list(mean + sem) + list(mean - sem)[::-1],
                fill='toself', fillcolor=_block_rgba(block, 0.2, scale),
                line=dict(width=0), mode='none', showlegend=False), row=1, col=1)

    # early-time changes: early-block non-probe (expected) minus late-block probe (unexpected)
    delta = psycho_or_chrono[:, :, 0, 0] - psycho_or_chrono[:, :, 1, 1]
    n_valid = np.sum(~np.isnan(delta), axis=0)
    mean_d = np.nanmean(delta, axis=0)
    sem_d = np.nanstd(delta, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))
    colour = _block_colour('early')

    fig.add_trace(go.Scatter(
        x=change_tfs, y=mean_d,
        mode='lines+markers',
        name='Early time: early-block minus late-block probe',
        line=dict(color=colour)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(change_tfs) + list(change_tfs)[::-1],
        y=list(mean_d + sem_d) + list(mean_d - sem_d)[::-1],
        fill='toself', fillcolor=_block_rgba('early', 0.2),
        line=dict(width=0), mode='none', showlegend=False), row=2, col=1)

    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                  row=2, col=1)

    fig.update_xaxes(title_text='Change TF', row=2, col=1)
    fig.update_yaxes(title_text=yax, row=1, col=1)
    fig.update_yaxes(title_text=delta_yax, row=2, col=1)
    fig.update_layout(template='plotly_white')
    return fig


def plot_elta(elta, config=ANALYSIS_OPTIONS):
    """average early-lick-triggered TF for each condition, with expectation delta"""
    n_samples = config.get('n_pre_lick_samples', 40)
    sample_rate = config.get('tf_sample_rate', 20)
    t = np.linspace(-n_samples / sample_rate, 0, n_samples)

    line_specs = {
        'earlyBlock_early': {'block': 'early', 'dash': 'solid',
                             'label': 'Early block, early lick'},
        'lateBlock_early': {'block': 'late', 'dash': 'dash',
                            'label': 'Late block, early lick'},
        'lateBlock_late': {'block': 'late', 'dash': 'solid',
                           'label': 'Late block, late lick'},
    }

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)

    for cond, spec in line_specs.items():
        if cond not in elta:
            continue
        scale = 0.65 if spec['dash'] == 'dash' else 1.0
        colour = _block_colour(spec['block'], scale)

        mean = elta[cond]['mean']
        sem = elta[cond]['sem']

        fig.add_trace(go.Scatter(
            x=t, y=mean, mode='lines', name=spec['label'],
            line=dict(color=colour, dash=spec['dash'], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(mean + sem) + list(mean - sem)[::-1],
            fill='toself', fillcolor=_block_rgba(spec['block'], 0.2, scale),
            line=dict(width=0), mode='none', showlegend=False), row=1, col=1)

    # delta: same time window (early-trial licks), early block minus late block
    if 'earlyBlock_early' in elta and 'lateBlock_early' in elta:
        e, l = elta['earlyBlock_early'], elta['lateBlock_early']
        e_subjs = e.get('subjs')
        l_subjs = l.get('subjs')
        if e_subjs is not None and l_subjs is not None:
            common = [s for s in e_subjs if s in l_subjs]
            e_idx = [e_subjs.index(s) for s in common]
            l_idx = [l_subjs.index(s) for s in common]
            delta = e['subj_means'][e_idx] - l['subj_means'][l_idx]
        else:
            delta = e['subj_means'] - l['subj_means']

        n_valid = np.sum(~np.isnan(delta), axis=0)
        mean_d = np.nanmean(delta, axis=0)
        sem_d = np.nanstd(delta, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))

        fig.add_trace(go.Scatter(
            x=t, y=mean_d, mode='lines',
            name='Early − Late block (early-trial licks)',
            line=dict(color='grey', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(mean_d + sem_d) + list(mean_d - sem_d)[::-1],
            fill='toself', fillcolor='rgba(128,128,128,0.2)',
            line=dict(width=0), mode='none', showlegend=False), row=2, col=1)

        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                      row=2, col=1)

    fig.update_xaxes(title_text='Time relative to lick (s)', row=2, col=1)
    fig.update_yaxes(title_text='Baseline stimulus (octaves)', row=1, col=1)
    fig.update_yaxes(title_text='delta stimulus (octaves)', row=2, col=1)
    fig.update_layout(template='plotly_white')
    return fig


PC_COLOURS = ['rgb(31,119,180)', 'rgb(255,127,14)', 'rgb(44,160,44)',
              'rgb(214,39,40)', 'rgb(148,103,189)']


def plot_eltc(eltc, config=ANALYSIS_OPTIONS, n_components=3, show_parallel=True):
    """PCA components and scree plots for each condition"""
    first_subj = next(v for v in eltc.values() if v)
    n_samples = next(iter(first_subj.values()))['components'].shape[1]
    sample_rate = config.get('tf_sample_rate', 20)
    t = np.linspace(-n_samples / sample_rate, 0, n_samples)

    block_map = {
        'earlyBlock_early': 'early',
        'lateBlock_early': 'late',
        'lateBlock_late': 'late',
    }

    cond_names = list(eltc.keys())
    fig = make_subplots(
        rows=len(cond_names), cols=2,
        column_widths=[0.65, 0.35],
        subplot_titles=[t for cond in cond_names
                        for t in (f'{cond} — Components', f'{cond} — Scree')])

    for r, cond in enumerate(cond_names):
        subj_data = eltc[cond]
        if not subj_data:
            continue

        block_colour = _block_colour(block_map[cond])

        for comp in range(n_components):
            all_comps = np.stack([subj_data[s]['components'][comp, :]
                                  for s in subj_data])
            mean_comp = np.mean(all_comps, axis=0)
            sem_comp = np.std(all_comps, axis=0) / np.sqrt(len(all_comps))

            pc_col = PC_COLOURS[comp % len(PC_COLOURS)]
            pc_rgb = pc_col.replace('rgb(', '').replace(')', '')

            fig.add_trace(go.Scatter(
                x=t, y=mean_comp, mode='lines', name=f'PC{comp + 1}',
                line=dict(color=pc_col, width=2),
                legendgroup=f'PC{comp + 1}', showlegend=(r == 0),
            ), row=r + 1, col=1)
            fig.add_trace(go.Scatter(
                x=list(t) + list(t)[::-1],
                y=list(mean_comp + sem_comp) + list(mean_comp - sem_comp)[::-1],
                fill='toself', fillcolor=f'rgba({pc_rgb},0.15)',
                line=dict(width=0), mode='none', showlegend=False,
                legendgroup=f'PC{comp + 1}',
            ), row=r + 1, col=1)

        n_pcs = min(v['explained_var'].shape[0] for v in subj_data.values())
        all_ev = np.stack([subj_data[s]['explained_var_ratio'][:n_pcs]
                           for s in subj_data])
        pc_labels = list(range(1, n_pcs + 1))

        for i in range(len(subj_data)):
            fig.add_trace(go.Scatter(
                x=pc_labels, y=all_ev[i], mode='lines+markers',
                line=dict(color=block_colour, width=1), marker=dict(size=4),
                opacity=0.15, showlegend=False,
            ), row=r + 1, col=2)

        mean_ev = np.mean(all_ev, axis=0)
        fig.add_trace(go.Scatter(
            x=pc_labels, y=mean_ev, mode='lines+markers',
            line=dict(color=block_colour, width=3), marker=dict(size=8),
            showlegend=False,
        ), row=r + 1, col=2)

        if show_parallel:
            all_parallel = np.stack([subj_data[s]['parallel_ev_thresh'][:n_pcs]
                                     for s in subj_data])
            mean_parallel = np.mean(all_parallel, axis=0)
            mean_n_sig = np.mean([subj_data[s]['n_sig'] for s in subj_data])

            fig.add_trace(go.Scatter(
                x=pc_labels, y=mean_parallel, mode='lines',
                name=f'{(1 - config["sig_thresh"]) * 100}% threshold',
                line=dict(color='grey', width=2, dash='dash'),
                showlegend=(r == 0),
            ), row=r + 1, col=2)
            fig.add_annotation(
                x=0.95, y=0.95,
                xref=f'x{r * 2 + 2} domain', yref=f'y{r * 2 + 2} domain',
                text=f'n_sig = {mean_n_sig:.1f}', showarrow=False,
                font=dict(size=12))

    for r in range(len(cond_names)):
        fig.update_xaxes(title_text='Time relative to lick (s)', row=r + 1, col=1)
        fig.update_yaxes(title_text='Loading', row=r + 1, col=1)
        fig.update_xaxes(title_text='Component', dtick=1, row=r + 1, col=2)
        fig.update_yaxes(title_text='% Variance Explained', row=r + 1, col=2)

    fig.update_layout(template='plotly_white',
                      height=350 * len(cond_names), width=950)
    return fig


def plot_eltc_comparison(eltc, config=ANALYSIS_OPTIONS, n_components=3):
    """each PC as a subplot, all conditions overlaid"""
    first_subj = next(v for v in eltc.values() if v)
    n_samples = next(iter(first_subj.values()))['components'].shape[1]
    sample_rate = config.get('tf_sample_rate', 20)
    t = np.linspace(-n_samples / sample_rate, 0, n_samples)

    line_specs = {
        'earlyBlock_early': {'block': 'early', 'dash': 'solid',
                             'label': 'Early block, early lick'},
        'lateBlock_early': {'block': 'late', 'dash': 'dash',
                            'label': 'Late block, early lick'},
        'lateBlock_late': {'block': 'late', 'dash': 'solid',
                           'label': 'Late block, late lick'},
    }

    fig = make_subplots(rows=1, cols=n_components,
                        subplot_titles=[f'PC{i + 1}' for i in range(n_components)])

    for cond, spec in line_specs.items():
        if cond not in eltc or not eltc[cond]:
            continue
        subj_data = eltc[cond]
        scale = 0.65 if spec['dash'] == 'dash' else 1.0
        colour = _block_colour(spec['block'], scale)

        for comp in range(n_components):
            all_comps = np.stack([subj_data[s]['components'][comp, :]
                                  for s in subj_data])
            mean_comp = np.mean(all_comps, axis=0)
            sem_comp = np.std(all_comps, axis=0) / np.sqrt(len(all_comps))

            fig.add_trace(go.Scatter(
                x=t, y=mean_comp, mode='lines', name=spec['label'],
                line=dict(color=colour, dash=spec['dash'], width=2),
                legendgroup=cond, showlegend=(comp == 0),
            ), row=1, col=comp + 1)
            fig.add_trace(go.Scatter(
                x=list(t) + list(t)[::-1],
                y=list(mean_comp + sem_comp) + list(mean_comp - sem_comp)[::-1],
                fill='toself', fillcolor=_block_rgba(spec['block'], 0.15, scale),
                line=dict(width=0), mode='none', showlegend=False,
                legendgroup=cond,
            ), row=1, col=comp + 1)

    for c in range(n_components):
        fig.update_xaxes(title_text='Time relative to lick (s)', row=1, col=c + 1)
        if c == 0:
            fig.update_yaxes(title_text='Loading', row=1, col=c + 1)

    fig.update_layout(template='plotly_white', height=400,
                      width=300 * n_components)
    return fig


def plot_el_hazard_rates(hazard_rates, config=ANALYSIS_OPTIONS):
    line_specs = {
        'early': {'n_key': 'early_n', 'dash': 'solid', 'label': 'Early block'},
        'late': {'n_key': 'late_n', 'dash': 'solid', 'label': 'Late block'},
    }

    min_n = config.get('min_hazard_samples', 100)
    change_wins = config.get('change_wins', {})

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)

    bin_centres = hazard_rates[next(iter(hazard_rates))]['binCentres']

    for block in ['early', 'late']:
        if block not in change_wins:
            continue
        win = change_wins[block]
        for xref, yref in [('x', 'y'), ('x2', 'y2')]:
            fig.add_shape(
                type='rect', x0=win[0], x1=win[1], y0=0, y1=1,
                xref=xref, yref=f'{yref} domain',
                fillcolor=_block_rgba(block, 0.12), line_width=0, layer='below')

    block_key_map = {'early': 'earlyBlock', 'late': 'lateBlock'}
    for block, spec in line_specs.items():
        colour = _block_colour(block)
        colour_rgba = _block_rgba(block, 0.15)

        all_rates = np.stack([hazard_rates[s][block_key_map[block]]
                              for s in hazard_rates])
        all_n = np.stack([hazard_rates[s][spec['n_key']] for s in hazard_rates])
        all_rates[:, all_n.sum(axis=0) < min_n] = np.nan

        n_valid = np.sum(~np.isnan(all_rates), axis=0).astype(float)
        mean_rate = np.nanmean(all_rates, axis=0)
        ci_rate = 1.96 * np.nanstd(all_rates, axis=0) / np.sqrt(n_valid)

        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean_rate + ci_rate,
            mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean_rate - ci_rate,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor=colour_rgba, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean_rate,
            mode='lines', name=spec['label'],
            line=dict(color=colour, width=2, dash=spec['dash'])), row=1, col=1)

    all_early = np.stack([hazard_rates[s]['earlyBlock'] for s in hazard_rates])
    all_late = np.stack([hazard_rates[s]['lateBlock'] for s in hazard_rates])
    all_n_early = np.stack([hazard_rates[s]['early_n'] for s in hazard_rates])
    all_n_late = np.stack([hazard_rates[s]['late_n'] for s in hazard_rates])

    insufficient = (all_n_early.sum(axis=0) < min_n) | (all_n_late.sum(axis=0) < min_n)
    all_early[:, insufficient] = np.nan
    all_late[:, insufficient] = np.nan

    deltas = all_early - all_late
    n_valid = np.sum(~np.isnan(deltas), axis=0).astype(float)
    mean_delta = np.nanmean(deltas, axis=0)
    ci_delta = 1.96 * np.nanstd(deltas, axis=0) / np.sqrt(n_valid)

    fig.add_trace(go.Scatter(
        x=bin_centres, y=mean_delta + ci_delta,
        mode='lines', line=dict(width=0), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bin_centres, y=mean_delta - ci_delta,
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(128,128,128,0.15)',
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bin_centres, y=mean_delta,
        mode='lines', name='Early - Late',
        line=dict(color='grey', width=2)), row=2, col=1)

    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'), row=2, col=1)

    fig.update_layout(template='plotly_white', width=600, height=500)
    fig.update_xaxes(title_text='Time in trial (s)', row=2, col=1)
    fig.update_yaxes(title_text='FA hazard rate', row=1, col=1)
    fig.update_yaxes(title_text='Delta hazard (Early - Late)', row=2, col=1)

    return fig


def plot_pulse_aligned_lick_prob(pulse_lick_prob, config=ANALYSIS_OPTIONS):
    """pulse-aligned lick probability, one figure per time window.
    returns {time_label: fig}"""
    first_subj = next(iter(pulse_lick_prob.values()))
    bin_centres = first_subj['binCentres']
    time_starts = first_subj['time_starts']
    time_win = first_subj['time_win']
    min_n = config.get('min_pulse_samples', 500)
    xlims = [-.6, .6]

    def get_probs_and_n(cond_key):
        subjs = [s for s in pulse_lick_prob if cond_key in pulse_lick_prob[s]]
        if not subjs:
            return None, None
        probs = np.stack([pulse_lick_prob[s][cond_key]['lickProb'] for s in subjs])
        n = np.stack([pulse_lick_prob[s][cond_key]['n'] for s in subjs])
        probs[:, n.sum(axis=0) < min_n] = np.nan
        return probs, n

    figs = {}
    for t_start in time_starts:
        t_end = t_start + time_win
        label = f'{t_start:.0f}-{t_end:.0f}s'

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.08)

        for block in ['early', 'late']:
            cond_key = f'{block}Block_{t_start:.0f}-{t_end:.0f}s'
            colour = _block_colour(block)
            colour_rgba = _block_rgba(block, 0.15)

            all_probs, _ = get_probs_and_n(cond_key)
            if all_probs is None:
                continue

            baseline = np.nanmean(all_probs, axis=1, keepdims=True)
            probs = all_probs - baseline

            n_valid = np.sum(~np.isnan(probs), axis=0).astype(float)
            mean_prob = np.nanmean(probs, axis=0)
            ci_prob = 1.96 * np.nanstd(probs, axis=0) / np.sqrt(
                np.where(n_valid > 0, n_valid, 1))

            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_prob + ci_prob,
                mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_prob - ci_prob,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=colour_rgba, showlegend=False),
                row=1, col=1)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_prob, mode='lines',
                name=f'{block.capitalize()} block',
                line=dict(color=colour, width=2)), row=1, col=1)

        early_key = f'earlyBlock_{t_start:.0f}-{t_end:.0f}s'
        late_key = f'lateBlock_{t_start:.0f}-{t_end:.0f}s'
        all_early, _ = get_probs_and_n(early_key)
        all_late, _ = get_probs_and_n(late_key)
        if all_early is not None and all_late is not None:
            early_bl = np.nanmean(all_early, axis=1, keepdims=True)
            late_bl = np.nanmean(all_late, axis=1, keepdims=True)
            deltas = (all_early - early_bl) - (all_late - late_bl)

            n_valid = np.sum(~np.isnan(deltas), axis=0).astype(float)
            mean_delta = np.nanmean(deltas, axis=0)
            ci_delta = 1.96 * np.nanstd(deltas, axis=0) / np.sqrt(
                np.where(n_valid > 0, n_valid, 1))

            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_delta + ci_delta,
                mode='lines', line=dict(width=0), showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_delta - ci_delta,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(128,128,128,0.15)',
                showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_delta, mode='lines', name='Early - Late',
                line=dict(color='grey', width=2)), row=2, col=1)

            fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                          row=2, col=1)

        fig.update_xaxes(range=xlims, row=1, col=1)
        fig.update_xaxes(range=xlims, title_text='Delta TF (oct)', row=2, col=1)
        fig.update_yaxes(title_text='Delta p(lick)', row=1, col=1)
        fig.update_yaxes(title_text='Early - Late', row=2, col=1)
        fig.update_layout(template='plotly_white', width=500, height=500,
                          title_text=label)
        figs[label] = fig

    return figs


def plot_pulse_lick_prob_2d(pulse_lick_prob, config=ANALYSIS_OPTIONS):
    """heatmap of P(lick) for TF(t) vs TF(t-lag), one figure per time window x lag.
    returns {label: fig}"""
    first_subj = next(iter(pulse_lick_prob.values()))
    bin_centres = first_subj['binCentres']
    time_starts = first_subj['time_starts']
    time_win = first_subj['time_win']
    min_n = config.get('min_pulse_samples_2d', 50)
    xlims = [-.6, .6]
    pulse_lags = config.get('tf_pulse_lags', [1, 2, 3, 4])

    frame_step = config.get('tf_sample_step', 3)
    frame_rate = config.get('frame_rate', 60)
    dt_ms = 1000 * frame_step / frame_rate

    subjs = list(pulse_lick_prob.keys())

    baseline_hw = config.get('baseline_half_width', 0.1)
    baseline_mask = np.abs(bin_centres) <= baseline_hw

    def _get_mean_2d(cond_key, lag):
        valid_subjs = [s for s in subjs if cond_key in pulse_lick_prob[s]]
        if not valid_subjs:
            return None, None
        all_2d = np.stack([pulse_lick_prob[s][cond_key]['lickProb2D'][lag]
                           for s in valid_subjs])
        all_n = np.stack([pulse_lick_prob[s][cond_key]['n2D'][lag]
                          for s in valid_subjs])
        low_n = all_n.sum(axis=0) < min_n
        all_2d[:, low_n] = np.nan
        return all_2d, valid_subjs

    def _smooth(mat):
        nan_mask = np.isnan(mat)
        out = mat.copy()
        out[nan_mask] = 0
        out = gaussian_filter(out, sigma=1, truncate=1.5)
        out[nan_mask] = np.nan
        return out

    def _baseline_1d(cond_key):
        """mean P(lick) near TF=0 per subject from the 1D curve, broadcastable to (n_subj, 1, 1)"""
        valid_subjs = [s for s in subjs if cond_key in pulse_lick_prob[s]]
        all_1d = np.stack([pulse_lick_prob[s][cond_key]['lickProb']
                           for s in valid_subjs])
        return np.nanmean(all_1d[:, baseline_mask], axis=1)[:, None, None]

    row_titles = ['Raw', 'Per-block norm', 'Global norm (late baseline)']
    n_rows = len(row_titles)

    # first pass on first time window only: get z-limits per row from 5th/95th percentiles
    t0 = time_starts[0]
    t0_end = t0 + time_win
    zlims = {r: [0, 1] for r in range(n_rows)}

    all_vals_by_row = {r: [] for r in range(n_rows)}
    for lag in pulse_lags:
        early_key = f'earlyBlock_{t0:.0f}-{t0_end:.0f}s'
        late_key = f'lateBlock_{t0:.0f}-{t0_end:.0f}s'
        all_early, _ = _get_mean_2d(early_key, lag)
        all_late, _ = _get_mean_2d(late_key, lag)
        if all_early is None or all_late is None:
            continue

        global_bl = _baseline_1d(late_key)
        block_data = {'early': (all_early, early_key), 'late': (all_late, late_key)}

        for row_idx in range(n_rows):
            for block in ['early', 'late']:
                ad, ckey = block_data[block]
                if row_idx == 0:
                    normed = ad
                elif row_idx == 1:
                    normed = ad - _baseline_1d(ckey)
                else:
                    normed = ad - global_bl
                mean_2d = _smooth(np.nanmean(normed, axis=0))
                vals = mean_2d[~np.isnan(mean_2d)]
                if len(vals) > 0:
                    all_vals_by_row[row_idx].append(vals)

    for r in range(n_rows):
        if all_vals_by_row[r]:
            pooled = np.concatenate(all_vals_by_row[r])
            if r == 0:
                zlims[r] = [np.percentile(pooled, 1), np.percentile(pooled, 99)]
            else:
                vmax = np.percentile(np.abs(pooled), 99)
                zlims[r] = [-vmax, vmax]

    # second pass: build all figures
    figs = {}
    for t_start in time_starts:
        t_end = t_start + time_win
        time_label = f'{t_start:.0f}-{t_end:.0f}s'

        for lag in pulse_lags:
            lag_ms = int(lag * dt_ms)
            label = f'{time_label}_lag{lag_ms}ms'

            early_key = f'earlyBlock_{t_start:.0f}-{t_end:.0f}s'
            late_key = f'lateBlock_{t_start:.0f}-{t_end:.0f}s'

            all_early, _ = _get_mean_2d(early_key, lag)
            all_late, _ = _get_mean_2d(late_key, lag)
            if all_early is None or all_late is None:
                continue

            global_bl = _baseline_1d(late_key)
            block_data = {'early': (all_early, early_key),
                          'late': (all_late, late_key)}

            fig = make_subplots(
                rows=3, cols=2, shared_xaxes=True, shared_yaxes=True,
                vertical_spacing=0.08, horizontal_spacing=0.08,
                subplot_titles=['Early block', 'Late block'] + [''] * 4,
                row_titles=row_titles)

            for row_idx in range(n_rows):
                caxis = f'coloraxis{row_idx + 1}' if row_idx > 0 else 'coloraxis'
                for col, block in enumerate(['early', 'late'], 1):
                    ad, ckey = block_data[block]
                    if row_idx == 0:
                        normed = ad
                    elif row_idx == 1:
                        normed = ad - _baseline_1d(ckey)
                    else:
                        normed = ad - global_bl

                    mean_2d = _smooth(np.nanmean(normed, axis=0))
                    fig.add_trace(go.Heatmap(
                        z=mean_2d, x=bin_centres, y=bin_centres,
                        coloraxis=caxis),
                        row=row_idx + 1, col=col)

                zlo, zhi = zlims[row_idx]
                if row_idx == 0:
                    cscale, cmid = 'Hot_r', None
                else:
                    cscale, cmid = 'RdBu_r', 0
                fig.layout[caxis] = dict(
                    colorscale=cscale,
                    cmin=zlo, cmax=zhi, cmid=cmid,
                    colorbar=dict(
                        title='P(lick)' if row_idx == 0 else 'dP(lick)',
                        len=0.27, y=1 - row_idx * 0.35))

            for col in [1, 2]:
                fig.update_xaxes(range=xlims, title_text='TF(t) oct',
                                 row=3, col=col)
            for row in [1, 2, 3]:
                fig.update_yaxes(range=xlims,
                                 title_text=f'TF(t-{lag_ms}ms) oct',
                                 row=row, col=1)

            fig.update_layout(
                template='plotly_white', width=700, height=900,
                title_text=f'{time_label}, lag = {lag_ms} ms')

            figs[label] = fig

    return figs
