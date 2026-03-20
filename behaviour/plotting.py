"""
plotly-based visualisation for behavioural analyses
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import uniform_filter1d

from config import BEHAVIOUR_PARAMS, PLOT_OPTIONS


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


def plot_psychometric(psycho_or_chrono, config=BEHAVIOUR_PARAMS):
    change_tfs = config['change_tfs']

    if np.nanmax(psycho_or_chrono.flatten()) > 1:
        yax = 'RT (s)'
    else:
        yax = 'P(Hit)'

    fig = go.Figure()
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
                line=dict(color=colour, dash=dash)))
            fig.add_trace(go.Scatter(
                x=list(change_tfs) + list(change_tfs)[::-1],
                y=list(mean + sem) + list(mean - sem)[::-1],
                fill='toself', fillcolor=_block_rgba(block, 0.2, scale),
                line=dict(width=0), mode='none', showlegend=False))

    fig.update_layout(xaxis_title='Change TF', yaxis_title=yax,
                      template='plotly_white')
    return fig


def plot_elta(elta, config=BEHAVIOUR_PARAMS):
    """average early-lick-triggered TF for each condition"""
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

    fig = go.Figure()
    for cond, spec in line_specs.items():
        if cond not in elta:
            continue
        scale = 0.65 if spec['dash'] == 'dash' else 1.0
        colour = _block_colour(spec['block'], scale)

        mean = elta[cond]['mean']
        sem = elta[cond]['sem']

        fig.add_trace(go.Scatter(
            x=t, y=mean, mode='lines', name=spec['label'],
            line=dict(color=colour, dash=spec['dash'], width=2)))
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(mean + sem) + list(mean - sem)[::-1],
            fill='toself', fillcolor=_block_rgba(spec['block'], 0.2, scale),
            line=dict(width=0), mode='none', showlegend=False))

    fig.update_layout(xaxis_title='Time relative to lick (s)',
                      yaxis_title='Baseline stimulus (octaves)',
                      template='plotly_white')
    return fig


PC_COLOURS = ['rgb(31,119,180)', 'rgb(255,127,14)', 'rgb(44,160,44)',
              'rgb(214,39,40)', 'rgb(148,103,189)']


def plot_eltc(eltc, config=BEHAVIOUR_PARAMS, n_components=3, show_parallel=True):
    """PCA components and scree plots for each condition"""
    n_samples = config.get('n_pre_lick_samples', 20)
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


def plot_eltc_comparison(eltc, config=BEHAVIOUR_PARAMS, n_components=3):
    """each PC as a subplot, all conditions overlaid"""
    n_samples = config.get('n_pre_lick_samples', 20)
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


def plot_el_hazard_rates(hazard_rates, config=BEHAVIOUR_PARAMS):
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


def plot_pulse_aligned_lick_prob(pulse_lick_prob, config=BEHAVIOUR_PARAMS):
    line_specs = {
        'early': {'cond_key': 'earlyBlock_earlyTrial', 'dash': 'solid',
                  'label': 'Early block'},
        'late': {'cond_key': 'lateBlock_earlyTrial', 'dash': 'solid',
                 'label': 'Late block'},
    }

    min_n = config.get('min_pulse_samples', 500)
    baseline_hw = config.get('baseline_half_width', 0.1)
    bin_centres = pulse_lick_prob[next(iter(pulse_lick_prob))]['binCentres']
    baseline_mask = np.abs(bin_centres) <= baseline_hw

    all_early_bl = np.stack([
        pulse_lick_prob[s]['lateBlock_earlyTrial']['lickProb']
        for s in pulse_lick_prob])
    all_n_bl = np.stack([
        pulse_lick_prob[s]['lateBlock_earlyTrial']['n']
        for s in pulse_lick_prob])
    all_early_bl[:, all_n_bl.sum(axis=0) < min_n] = np.nan
    subject_baselines = np.nanmean(all_early_bl[:, baseline_mask], axis=1, keepdims=True)

    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.08, horizontal_spacing=0.12,
        subplot_titles=['Global norm', 'Per-block norm', '', ''])

    def get_probs_and_n(cond_key):
        probs = np.stack([pulse_lick_prob[s][cond_key]['lickProb']
                          for s in pulse_lick_prob])
        n = np.stack([pulse_lick_prob[s][cond_key]['n']
                      for s in pulse_lick_prob])
        probs[:, n.sum(axis=0) < min_n] = np.nan
        return probs, n

    for block, spec in line_specs.items():
        colour = _block_colour(block)
        colour_rgba = _block_rgba(block, 0.15)

        all_probs, _ = get_probs_and_n(spec['cond_key'])

        probs_global = all_probs - subject_baselines
        probs_local = all_probs - np.nanmean(all_probs[:, baseline_mask],
                                              axis=1, keepdims=True)

        for col, probs in [(1, probs_global), (2, probs_local)]:
            n_valid = np.sum(~np.isnan(probs), axis=0).astype(float)
            mean_prob = np.nanmean(probs, axis=0)
            ci_prob = 1.96 * np.nanstd(probs, axis=0) / np.sqrt(n_valid)

            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_prob + ci_prob,
                mode='lines', line=dict(width=0), showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_prob - ci_prob,
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor=colour_rgba, showlegend=False),
                row=1, col=col)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_prob, mode='lines', name=spec['label'],
                line=dict(color=colour, width=2, dash=spec['dash']),
                showlegend=(col == 1)), row=1, col=col)

    all_early, all_n_early = get_probs_and_n('earlyBlock_earlyTrial')
    all_late, all_n_late = get_probs_and_n('lateBlock_earlyTrial')

    insufficient = (all_n_early.sum(axis=0) < min_n) | (all_n_late.sum(axis=0) < min_n)
    all_early[:, insufficient] = np.nan
    all_late[:, insufficient] = np.nan

    baseline_early = np.nanmean(all_early[:, baseline_mask], axis=1, keepdims=True)
    baseline_late = np.nanmean(all_late[:, baseline_mask], axis=1, keepdims=True)

    pairs = {
        1: (all_early - subject_baselines, all_late - subject_baselines),
        2: (all_early - baseline_early, all_late - baseline_late),
    }

    for col, (early_norm, late_norm) in pairs.items():
        deltas = early_norm - late_norm
        n_valid = np.sum(~np.isnan(deltas), axis=0).astype(float)
        mean_delta = np.nanmean(deltas, axis=0)
        ci_delta = 1.96 * np.nanstd(deltas, axis=0) / np.sqrt(n_valid)

        for i in range(len(deltas)):
            fig.add_trace(go.Scatter(
                x=bin_centres, y=uniform_filter1d(deltas[i], size=5),
                mode='lines', line=dict(color='rgba(128,128,128,0.3)', width=0.75),
                showlegend=False), row=2, col=col)

        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean_delta + ci_delta,
            mode='lines', line=dict(width=0), showlegend=False), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean_delta - ci_delta,
            mode='lines', line=dict(width=0),
            fill='tonexty', fillcolor='rgba(128,128,128,0.15)',
            showlegend=False), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean_delta, mode='lines', name='Early - Late',
            line=dict(color='grey', width=2), showlegend=(col == 1)), row=2, col=col)

        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                      row=2, col=col)

    xlims = [-.6, .6]
    fig.update_layout(template='plotly_white', width=600, height=600)
    for col in [1, 2]:
        fig.update_xaxes(range=xlims, row=1, col=col)
        fig.update_xaxes(range=xlims, row=2, col=col)
        fig.update_xaxes(title_text='Delta TF (oct)', row=2, col=col)
    fig.update_yaxes(title_text='Delta p(lick) from baseline', row=1, col=1)
    fig.update_yaxes(title_text='Delta p(lick) (Early - Late)', row=2, col=1)

    return fig
