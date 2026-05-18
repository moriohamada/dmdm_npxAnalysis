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

    fig.add_trace(go.Scatter(
        x=change_tfs, y=mean_d,
        mode='lines+markers',
        name='Early time: early-block minus late-block probe',
        line=dict(color='grey')), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=list(change_tfs) + list(change_tfs)[::-1],
        y=list(mean_d + sem_d) + list(mean_d - sem_d)[::-1],
        fill='toself', fillcolor='rgba(128,128,128,0.2)',
        line=dict(width=0), mode='none', showlegend=False), row=2, col=1)

    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                  row=2, col=1)

    fig.update_xaxes(title_text='Change TF', row=2, col=1)
    fig.update_yaxes(title_text=yax, row=1, col=1)
    fig.update_yaxes(title_text=delta_yax, row=2, col=1)
    fig.update_layout(template='plotly_white')
    return fig


def plot_psychometric_fits(params, n_hits, n_trials, changes=None,
                            config=ANALYSIS_OPTIONS):
    """
    psychometric data + fitted curves per block, with early-late delta below.

    Inputs:
        params: dict {block: param arrays} from _fit_psychometric per block.
        n_hits, n_trials: 4D arrays (n_animals, n_changes, 2 block, 2 probe).
            sliced internally to early-block non-probe and late-block probe.
        changes: (n_chs,) Hz-above-baseline; defaults to config['change_tfs'] - 1.
    """
    if changes is None:
        changes = np.asarray(config['change_tfs']) - 1
    changes = np.asarray(changes)

    # diagonal slices: early-block non-probe vs late-block probe (matched in-trial timing)
    block_idx = {'early': np.s_[:, :, 0, 0], 'late': np.s_[:, :, 1, 1]}

    x_grid = np.linspace(changes.min(), changes.max(), 200)

    def p_hit(alpha, beta, gamma, lapse, x):
        ratio = np.clip(x / max(alpha, 1e-9), 0, 50) ** beta
        return gamma + (1 - gamma - lapse) * (1 - np.exp(-ratio))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)

    # per-animal curves stacked into (n_subj, n_grid) for the delta plot
    block_curves = {}

    for block in ('early', 'late'):
        if block not in params:
            continue
        prm = params[block]
        nh = n_hits[block_idx[block]]
        nt = n_trials[block_idx[block]]
        bold = _block_colour(block)
        faint = _block_rgba(block, 0.25)

        n_subj = nh.shape[0]
        curves = np.full((n_subj, len(x_grid)), np.nan)
        for i in range(n_subj):
            valid = nt[i] > 0
            if valid.sum() > 0:
                rate = nh[i, valid] / nt[i, valid]
                fig.add_trace(go.Scatter(
                    x=changes[valid], y=rate, mode='markers',
                    marker=dict(color=faint, size=5),
                    showlegend=False), row=1, col=1)

            if np.isnan(prm['alpha'][i]):
                continue
            curves[i] = p_hit(prm['alpha'][i], prm['beta'][i],
                              prm['gamma'][i], prm['lapse'][i], x_grid)
            fig.add_trace(go.Scatter(
                x=x_grid, y=curves[i], mode='lines',
                line=dict(color=faint, width=1),
                showlegend=False), row=1, col=1)

        ok = ~np.isnan(curves[:, 0])
        if ok.any():
            mean_curve = np.nanmean(curves[ok], axis=0)
            fig.add_trace(go.Scatter(
                x=x_grid, y=mean_curve, mode='lines', name=block,
                line=dict(color=bold, width=3)), row=1, col=1)

        block_curves[block] = curves

    # delta: paired per animal, mean and SEM across animals
    if 'early' in block_curves and 'late' in block_curves:
        delta = block_curves['early'] - block_curves['late']
        ok = ~np.isnan(delta[:, 0])
        n_valid = ok.sum()
        if n_valid > 0:
            d_mean = np.nanmean(delta[ok], axis=0)
            d_ci = 1.96 * np.nanstd(delta[ok], axis=0) / np.sqrt(max(n_valid, 1))
            fig.add_trace(go.Scatter(
                x=x_grid, y=d_mean, mode='lines',
                name='early - late', line=dict(color='grey', width=2)),
                row=2, col=1)
            fig.add_trace(go.Scatter(
                x=list(x_grid) + list(x_grid)[::-1],
                y=list(d_mean + d_ci) + list(d_mean - d_ci)[::-1],
                fill='toself', fillcolor='rgba(128,128,128,0.2)',
                line=dict(width=0), mode='none', showlegend=False),
                row=2, col=1)

        # per-animal empirical delta points at each change tf
        rate_E = np.where(n_trials[block_idx['early']] > 0,
                          n_hits[block_idx['early']]
                          / np.maximum(n_trials[block_idx['early']], 1), np.nan)
        rate_L = np.where(n_trials[block_idx['late']] > 0,
                          n_hits[block_idx['late']]
                          / np.maximum(n_trials[block_idx['late']], 1), np.nan)
        emp_delta = rate_E - rate_L
        for i in range(emp_delta.shape[0]):
            valid = ~np.isnan(emp_delta[i])
            if valid.any():
                fig.add_trace(go.Scatter(
                    x=changes[valid], y=emp_delta[i, valid],
                    mode='markers',
                    marker=dict(color='rgba(128,128,128,0.45)', size=5),
                    showlegend=False), row=2, col=1)

        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                      row=2, col=1)

    tickvals = changes.tolist()
    ticktext = [f'{c + 1:g}' for c in changes]
    fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=0,
                     row=1, col=1)
    fig.update_xaxes(title_text='Change TF (Hz)',
                     tickvals=tickvals, ticktext=ticktext, tickangle=0,
                     row=2, col=1)
    fig.update_yaxes(title_text='P(Hit)', range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text='delta P(Hit)', row=2, col=1)
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
            name='Early - Late block (early-trial licks)',
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


def plot_lts_quant(lts_quant, config=ANALYSIS_OPTIONS, alpha=0.05):
    """
    ELTA and ELTV per block (earlyBlock_early vs lateBlock_late), mean ± SEM
    across animals. ticks below each panel mark timepoints where the paired
    test reaches p < alpha.
    """
    n_samples = lts_quant['elta_E'].shape[1]
    sample_rate = config.get('tf_sample_rate', 20)
    t = np.linspace(-n_samples / sample_rate, 0, n_samples)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=['ELTA (mean stim)', 'ELTV (variance)'])

    def add_band(row, data, block):
        mean = data.mean(axis=0)
        sem = data.std(axis=0) / np.sqrt(len(data))
        colour = _block_colour(block)
        fig.add_trace(go.Scatter(
            x=t, y=mean, mode='lines',
            name=f'{block.capitalize()} block',
            line=dict(color=colour, width=2),
            legendgroup=block, showlegend=(row == 1)), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(mean + sem) + list(mean - sem)[::-1],
            fill='toself', fillcolor=_block_rgba(block, 0.2),
            line=dict(width=0), mode='none', showlegend=False,
            legendgroup=block), row=row, col=1)

    add_band(1, lts_quant['elta_E'], 'early')
    add_band(1, lts_quant['elta_L'], 'late')
    add_band(2, lts_quant['eltv_E'], 'early')
    add_band(2, lts_quant['eltv_L'], 'late')

    for row, stat_key, d_E, d_L in [
        (1, 'mean_stats', lts_quant['elta_E'], lts_quant['elta_L']),
        (2, 'var_stats',  lts_quant['eltv_E'], lts_quant['eltv_L']),
    ]:
        pvals = lts_quant[stat_key][:, 1]
        sig_t = t[pvals < alpha]
        if len(sig_t):
            y_min = min(d_E.min(), d_L.min())
            y_max = max(d_E.max(), d_L.max())
            y_tick = y_min - 0.05 * (y_max - y_min)
            fig.add_trace(go.Scatter(
                x=sig_t, y=np.full(len(sig_t), y_tick),
                mode='markers',
                marker=dict(symbol='line-ns-open', color='black', size=8),
                showlegend=False), row=row, col=1)

    fig.update_xaxes(title_text='Time relative to lick (s)', row=2, col=1)
    fig.update_yaxes(title_text='Stim (octaves)', row=1, col=1)
    fig.update_yaxes(title_text='Variance (oct^2)', row=2, col=1)
    fig.update_layout(template='plotly_white', height=600, width=600)
    return fig


def plot_lts_pcs(lts_quant, config=ANALYSIS_OPTIONS):
    """
    common-PC summary from quantify_lick_triggered_stim:
    row 1: mean PC loading across animals (one panel per PC)
    row 2: per-animal projection mean for E vs L
    row 3: per-animal projection variance for E vs L
    p-values from paired test in subplot titles
    """
    n_components, n_samples = lts_quant['pc_components'].shape
    sample_rate = config.get('tf_sample_rate', 20)
    t = np.linspace(-n_samples / sample_rate, 0, n_samples)

    titles = (
        [f'PC{c + 1} loading' for c in range(n_components)] +
        [f'PC{c + 1} mean (p={lts_quant["projection_mean_stats"][c][1]:.3f})'
         for c in range(n_components)] +
        [f'PC{c + 1} var (p={lts_quant["projection_var_stats"][c][1]:.3f})'
         for c in range(n_components)]
    )

    fig = make_subplots(rows=3, cols=n_components,
                        subplot_titles=titles, vertical_spacing=0.13)

    def paired_strip(vals_E, vals_L, row, col):
        for i in range(len(vals_E)):
            fig.add_trace(go.Scatter(
                x=['early', 'late'], y=[vals_E[i], vals_L[i]],
                mode='lines+markers',
                line=dict(color='grey', width=1),
                marker=dict(size=4, color='grey'),
                opacity=0.5, showlegend=False), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=['early', 'late'], y=[vals_E.mean(), vals_L.mean()],
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=10, color='black'),
            showlegend=False), row=row, col=col)

    for c in range(n_components):
        fig.add_trace(go.Scatter(
            x=t, y=lts_quant['pc_components'][c], mode='lines',
            line=dict(color='black', width=3),
            showlegend=False), row=1, col=c + 1)
        fig.add_hline(y=0, line=dict(color='grey', width=1, dash='dot'),
                      row=1, col=c + 1)

        paired_strip(lts_quant['projection_mean_E'][:, c],
                     lts_quant['projection_mean_L'][:, c], row=2, col=c + 1)
        paired_strip(lts_quant['projection_var_E'][:, c],
                     lts_quant['projection_var_L'][:, c], row=3, col=c + 1)

    for c in range(n_components):
        fig.update_xaxes(title_text='Time rel lick (s)', row=1, col=c + 1)
        if c == 0:
            fig.update_yaxes(title_text='Loading', row=1, col=1)
            fig.update_yaxes(title_text='Projection mean', row=2, col=1)
            fig.update_yaxes(title_text='Projection variance', row=3, col=1)

    fig.update_layout(template='plotly_white',
                      height=850, width=320 * n_components)
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
                              for s in hazard_rates]).astype(float)
        all_n = np.stack([hazard_rates[s][spec['n_key']] for s in hazard_rates])
        all_rates[all_n < min_n] = np.nan

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

    all_early = np.stack([hazard_rates[s]['earlyBlock'] for s in hazard_rates]).astype(float)
    all_late = np.stack([hazard_rates[s]['lateBlock'] for s in hazard_rates]).astype(float)
    all_n_early = np.stack([hazard_rates[s]['early_n'] for s in hazard_rates])
    all_n_late = np.stack([hazard_rates[s]['late_n'] for s in hazard_rates])

    insufficient = (all_n_early < min_n) | (all_n_late < min_n)
    all_early[insufficient] = np.nan
    all_late[insufficient] = np.nan

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


def plot_hazard_rate_stats(hazard_stats):
    """3-row plot of bin-wise hazard rate stats from quantify_hazard_rates:
    row 1: mean early/late hazard (per-block valid bins)
    row 2: mean early-late delta (both-block valid bins)
    row 3: bin-wise p-values on log scale, with 0.01/.05 references"""
    bin_centres = hazard_stats['binCentres']
    early = hazard_stats['early']
    late  = hazard_stats['late']
    diffs = hazard_stats['diffs']
    ps    = hazard_stats['ps']

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.3, 0.25],
                        vertical_spacing=0.06)

    for block, vals in (('early', early), ('late', late)):
        colour = _block_colour(block)
        colour_rgba = _block_rgba(block, 0.15)
        n_valid = np.sum(~np.isnan(vals), axis=0).astype(float)
        mean = np.nanmean(vals, axis=0)
        ci = 1.96 * np.nanstd(vals, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))

        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean + ci, mode='lines',
            line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean - ci, mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor=colour_rgba,
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=bin_centres, y=mean, mode='lines',
            name=f'{block.capitalize()} block',
            line=dict(color=colour, width=2)), row=1, col=1)

    n_valid = np.sum(~np.isnan(diffs), axis=0).astype(float)
    mean_d = np.nanmean(diffs, axis=0)
    ci_d = 1.96 * np.nanstd(diffs, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))

    fig.add_trace(go.Scatter(
        x=bin_centres, y=mean_d + ci_d, mode='lines',
        line=dict(width=0), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bin_centres, y=mean_d - ci_d, mode='lines',
        line=dict(width=0), fill='tonexty',
        fillcolor='rgba(128,128,128,0.18)', showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=bin_centres, y=mean_d, mode='lines',
        name='Early - Late', line=dict(color='grey', width=2)), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=bin_centres, y=ps, mode='lines',
        line=dict(color='black', width=1.5), showlegend=False), row=3, col=1)
    fig.add_hline(y=0.01, line=dict(color='grey', width=1, dash='dash'), row=3, col=1)
    fig.add_hline(y=0.05, line=dict(color='grey', width=1, dash='dash'), row=3, col=1)

    fig.update_yaxes(type='log', row=3, col=1,
                     title_text=f'p ({hazard_stats["sig_test"]})')
    fig.update_yaxes(title_text='FA hazard rate', row=1, col=1)
    fig.update_yaxes(title_text='Early - Late', row=2, col=1)
    fig.update_xaxes(title_text='Time in trial (s)', row=3, col=1)
    fig.update_layout(template='plotly_white', width=600, height=700)

    return fig


def plot_pulse_lick_fits(pulse_fits, min_animals: int = 5):
    """3-row x 2-col plot of OLS fits from quantify_pulse_lick_probability.
    col 1: bias; col 2: slope.
    row 1: early vs late, per-animal thin + group mean +- 95% CI
    row 2: paired delta (early - late), per-animal thin + group mean +- 95% CI
    row 3: log-scale p-values vs 0.01 reference

    time points with fewer than min_animals valid fits (in either block for the
    means; in the paired set for delta/p) are dropped entirely.
    """
    time_starts = np.asarray(pulse_fits['time_starts'], dtype=float)
    time_win = pulse_fits['time_win']
    by_win = pulse_fits['by_window']
    labels = [f'{t:.0f}-{t + time_win:.0f}s' for t in time_starts]
    x = time_starts + time_win / 2

    fig = make_subplots(rows=3, cols=2, shared_xaxes=True,
                        row_heights=[0.4, 0.35, 0.25],
                        vertical_spacing=0.06, horizontal_spacing=0.12,
                        column_titles=['Bias (intercept)',
                                       'Slope (per TF oct)'])

    for col, param in enumerate(('bias', 'slope'), start=1):
        block_vals = {}
        for block in ('early', 'late'):
            colour = _block_colour(block)
            colour_rgba = _block_rgba(block, 0.15)
            vals = np.stack([by_win[lab][block][param] for lab in labels],
                            axis=1).astype(float)
            sparse = np.sum(~np.isnan(vals), axis=0) < min_animals
            vals[:, sparse] = np.nan
            block_vals[block] = vals

            n_valid = np.sum(~np.isnan(vals), axis=0).astype(float)
            mean = np.nanmean(vals, axis=0)
            ci = 1.96 * np.nanstd(vals, axis=0) / np.sqrt(
                np.where(n_valid > 0, n_valid, 1))

            for i in range(vals.shape[0]):
                fig.add_trace(go.Scatter(
                    x=x, y=vals[i], mode='lines',
                    line=dict(color=colour_rgba, width=0.75),
                    showlegend=False), row=1, col=col)

            fig.add_trace(go.Scatter(
                x=x, y=mean + ci, mode='lines',
                line=dict(width=0), showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=x, y=mean - ci, mode='lines',
                line=dict(width=0), fill='tonexty',
                fillcolor=colour_rgba, showlegend=False), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode='lines+markers',
                name=f'{block.capitalize()} block',
                line=dict(color=colour, width=2),
                showlegend=(col == 1)), row=1, col=col)

        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                      row=1, col=col)

        deltas = block_vals['early'] - block_vals['late']
        n_valid = np.sum(~np.isnan(deltas), axis=0).astype(float)
        sparse_d = n_valid < min_animals
        deltas[:, sparse_d] = np.nan
        n_valid = np.sum(~np.isnan(deltas), axis=0).astype(float)
        d_mean = np.nanmean(deltas, axis=0)
        d_ci = 1.96 * np.nanstd(deltas, axis=0) / np.sqrt(
            np.where(n_valid > 0, n_valid, 1))

        for i in range(deltas.shape[0]):
            fig.add_trace(go.Scatter(
                x=x, y=deltas[i], mode='lines',
                line=dict(color='rgba(128,128,128,0.3)', width=0.75),
                showlegend=False), row=2, col=col)

        fig.add_trace(go.Scatter(
            x=x, y=d_mean + d_ci, mode='lines',
            line=dict(width=0), showlegend=False), row=2, col=col)
        fig.add_trace(go.Scatter(
            x=x, y=d_mean - d_ci, mode='lines',
            line=dict(width=0), fill='tonexty',
            fillcolor='rgba(128,128,128,0.18)', showlegend=False),
            row=2, col=col)
        fig.add_trace(go.Scatter(
            x=x, y=d_mean, mode='lines+markers',
            name='Early - Late', line=dict(color='grey', width=2),
            showlegend=(col == 1)), row=2, col=col)
        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                      row=2, col=col)

        ps = np.array([by_win[lab]['stats'][param][1] for lab in labels])
        ps = np.where(sparse_d, np.nan, ps)
        fig.add_trace(go.Scatter(
            x=x, y=ps, mode='lines+markers',
            line=dict(color='black', width=1.5),
            marker=dict(size=5), showlegend=False), row=3, col=col)
        fig.add_hline(y=0.01, line=dict(color='grey', width=1, dash='dash'),
                      row=3, col=col)
        fig.update_yaxes(type='log', row=3, col=col)
        fig.update_xaxes(title_text='Time in trial (s, window centre)',
                         row=3, col=col)

    fig.update_yaxes(title_text='Block mean', row=1, col=1)
    fig.update_yaxes(title_text='Early - Late', row=2, col=1)
    fig.update_yaxes(title_text=f'p ({pulse_fits["sig_test"]})', row=3, col=1)
    fig.update_layout(template='plotly_white', width=1000, height=800)

    return fig


def plot_integration_time(int_quant, alpha=0.05):
    """quantification of integration time from two-pulse interaction index J:
    left panel: J(delay) per block (mean +- SEM, per-animal thin lines, sig
                ticks where one-sample test J > 0 reaches p < alpha;
                integration time = largest delay with sig J > 0).
    right panel: paired tau (exp-decay timescale) per animal, E vs L."""
    delay_centres = np.asarray(int_quant['delay_centres'])

    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35],
                        subplot_titles=['J(delay)', 'Decay tau'])

    # left: J vs delay per block
    for block in ('early', 'late'):
        b = int_quant[block]
        colour = _block_colour(block)
        colour_rgba = _block_rgba(block, 0.15)

        for j_animal in b['J']:
            fig.add_trace(go.Scatter(
                x=delay_centres, y=j_animal, mode='lines',
                line=dict(color=colour_rgba, width=0.75),
                showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=delay_centres, y=b['J_mean'] + b['J_sem'], mode='lines',
            line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=delay_centres, y=b['J_mean'] - b['J_sem'], mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor=colour_rgba,
            showlegend=False), row=1, col=1)
        int_t = b['integration_time']
        int_label = f'{int_t:.2f}s' if not np.isnan(int_t) else 'n.s.'
        fig.add_trace(go.Scatter(
            x=delay_centres, y=b['J_mean'], mode='lines+markers',
            name=f'{block.capitalize()} block (int. time = {int_label})',
            line=dict(color=colour, width=2),
            marker=dict(size=5)), row=1, col=1)

    # significance ticks per block, stacked under the curves
    y_concat = np.concatenate([int_quant['early']['J_mean'],
                               int_quant['late']['J_mean']])
    finite = y_concat[np.isfinite(y_concat)]
    if finite.size:
        y_lo, y_hi = float(finite.min()), float(finite.max())
        span = max(y_hi - y_lo, 1e-3)
        for k, block in enumerate(('early', 'late')):
            b = int_quant[block]
            sig_pos = (b['pval'] < alpha) & (b['J_mean'] > 0)
            if not sig_pos.any():
                continue
            tick_y = y_lo - (0.05 + 0.05 * k) * span
            fig.add_trace(go.Scatter(
                x=delay_centres[sig_pos],
                y=np.full(int(sig_pos.sum()), tick_y),
                mode='markers',
                marker=dict(symbol='line-ns-open',
                            color=_block_colour(block), size=8),
                showlegend=False), row=1, col=1)

    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                  row=1, col=1)

    # right: paired tau strip plot
    eT = np.asarray(int_quant['early']['tau'])
    lT = np.asarray(int_quant['late']['tau'])
    paired = ~np.isnan(eT) & ~np.isnan(lT)
    e_paired, l_paired = eT[paired], lT[paired]

    for i in range(len(e_paired)):
        fig.add_trace(go.Scatter(
            x=['early', 'late'], y=[e_paired[i], l_paired[i]],
            mode='lines+markers',
            line=dict(color='grey', width=1),
            marker=dict(size=5, color='grey'),
            opacity=0.5, showlegend=False), row=1, col=2)
    if len(e_paired):
        fig.add_trace(go.Scatter(
            x=['early', 'late'],
            y=[e_paired.mean(), l_paired.mean()],
            mode='lines+markers',
            line=dict(color='black', width=3),
            marker=dict(size=10, color='black'),
            showlegend=False), row=1, col=2)

    tau_p = int_quant['tau_stats'][1]
    p_str = f'{tau_p:.3f}' if not np.isnan(tau_p) else 'n/a'
    fig.layout.annotations[1].text = f'Decay tau (p = {p_str})'

    fig.update_xaxes(title_text='Inter-pulse delay (s)', row=1, col=1)
    fig.update_yaxes(title_text='Interaction index J', row=1, col=1)
    fig.update_yaxes(title_text='tau (s)', row=1, col=2)
    fig.update_layout(template='plotly_white', width=900, height=400)

    return fig


def plot_pulse_aligned_lick_prob(pulse_lick_prob, config=ANALYSIS_OPTIONS):
    """pulse-aligned lick probability, one figure per time window.
    cols: per-block norm (subtract own near-TF=0 baseline),
          global norm (subtract late-block near-TF=0 baseline from both).
    shading is 95% CI across animals. delta row shows individual animals as
    thin smoothed traces under the group mean.
    returns {time_label: fig}"""
    first_subj = next(iter(pulse_lick_prob.values()))
    bin_centres = first_subj['binCentres']
    time_starts = first_subj['time_starts']
    time_win = first_subj['time_win']
    min_n = config.get('min_pulse_samples', 500)
    baseline_hw = config.get('baseline_half_width', 0.1)
    baseline_mask = np.abs(bin_centres) <= baseline_hw
    xlims = [-.6, .6]
    col_titles = ['Per-block norm', 'Global norm (late baseline)']

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

        early_key = f'earlyBlock_{t_start:.0f}-{t_end:.0f}s'
        late_key  = f'lateBlock_{t_start:.0f}-{t_end:.0f}s'
        all_early, _ = get_probs_and_n(early_key)
        all_late,  _ = get_probs_and_n(late_key)
        if all_early is None or all_late is None:
            continue

        early_bl = np.nanmean(all_early[:, baseline_mask], axis=1, keepdims=True)
        late_bl  = np.nanmean(all_late[:,  baseline_mask], axis=1, keepdims=True)

        # (early_normed, late_normed) per column
        normed = {
            1: (all_early - early_bl, all_late - late_bl),  # per-block
            2: (all_early - late_bl,  all_late - late_bl),  # global
        }

        fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                            row_heights=[0.6, 0.4],
                            vertical_spacing=0.08, horizontal_spacing=0.12,
                            column_titles=col_titles)

        for col, (early_n, late_n) in normed.items():
            for block, probs in (('early', early_n), ('late', late_n)):
                colour = _block_colour(block)
                colour_rgba = _block_rgba(block, 0.15)
                n_valid = np.sum(~np.isnan(probs), axis=0).astype(float)
                mean_p = np.nanmean(probs, axis=0)
                ci_p = 1.96 * np.nanstd(probs, axis=0) / np.sqrt(
                    np.where(n_valid > 0, n_valid, 1))

                fig.add_trace(go.Scatter(
                    x=bin_centres, y=mean_p + ci_p, mode='lines',
                    line=dict(width=0), showlegend=False), row=1, col=col)
                fig.add_trace(go.Scatter(
                    x=bin_centres, y=mean_p - ci_p, mode='lines',
                    line=dict(width=0), fill='tonexty',
                    fillcolor=colour_rgba, showlegend=False), row=1, col=col)
                fig.add_trace(go.Scatter(
                    x=bin_centres, y=mean_p, mode='lines',
                    name=f'{block.capitalize()} block',
                    line=dict(color=colour, width=2),
                    showlegend=(col == 1)), row=1, col=col)

            deltas = early_n - late_n
            n_valid = np.sum(~np.isnan(deltas), axis=0).astype(float)
            mean_d = np.nanmean(deltas, axis=0)
            ci_d = 1.96 * np.nanstd(deltas, axis=0) / np.sqrt(
                np.where(n_valid > 0, n_valid, 1))

            for i in range(len(deltas)):
                fig.add_trace(go.Scatter(
                    x=bin_centres, y=uniform_filter1d(deltas[i], size=5),
                    mode='lines',
                    line=dict(color='rgba(128,128,128,0.3)', width=0.75),
                    showlegend=False), row=2, col=col)

            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_d + ci_d, mode='lines',
                line=dict(width=0), showlegend=False), row=2, col=col)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_d - ci_d, mode='lines',
                line=dict(width=0), fill='tonexty',
                fillcolor='rgba(128,128,128,0.15)', showlegend=False),
                row=2, col=col)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean_d, mode='lines', name='Early - Late',
                line=dict(color='grey', width=2),
                showlegend=(col == 1)), row=2, col=col)
            fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                          row=2, col=col)

        for col in (1, 2):
            fig.update_xaxes(range=xlims, row=1, col=col)
            fig.update_xaxes(range=xlims, title_text='Delta TF (oct)',
                             row=2, col=col)
        fig.update_yaxes(title_text='Delta P(lick) from baseline', row=1, col=1)
        fig.update_yaxes(title_text='Delta P(lick) (Early - Late)',
                         row=2, col=1)
        fig.update_layout(template='plotly_white', width=900, height=550,
                          title_text=label)
        figs[label] = fig

    return figs


def plot_pulse_lick_prob_by_period(pulse_lick_prob,
                                   config=ANALYSIS_OPTIONS):
    """pulse-aligned lick probability for three trial-period conditions,
    shown under raw, per-block-normalised, and global-normalised scaling.

    conditions:
        earlyBlock_early: early block, early in trial
        lateBlock_early:  late block, early in trial (probe-like)
        lateBlock_late:   late block, late in trial

    trial-period split uses the project-wide convention:
        early-in-trial: ignore_trial_start -> tr_split_time
        late-in-trial:  tr_split_time -> end of data
    saved sliding windows that fall fully inside each range are pooled
    (n-weighted across windows).
    """
    first_subj = next(iter(pulse_lick_prob.values()))
    bin_centres = first_subj['binCentres']
    time_starts = first_subj['time_starts']
    time_win = first_subj['time_win']
    min_n = config.get('min_pulse_samples', 500)
    xlims = [-.6, .6]
    baseline_hw = config.get('baseline_half_width', 0.1)
    baseline_mask = np.abs(bin_centres) <= baseline_hw

    t_split = config['tr_split_time']
    t_early_start = config['ignore_trial_start']
    t_late_end = float(time_starts[-1] + time_win)

    def windows_in_range(t_lo, t_hi):
        """sliding-window keys whose [start, start+time_win] fits in [t_lo, t_hi]"""
        return [t for t in time_starts
                if t >= t_lo and t + time_win <= t_hi]

    early_starts = windows_in_range(t_early_start, t_split)
    late_starts = windows_in_range(t_split, t_late_end)

    period_starts = {'early': early_starts, 'late': late_starts}

    cond_specs = {
        'earlyBlock_early': ('early', 'early', 'solid'),
        'lateBlock_early':  ('late',  'early', 'dash'),
        'lateBlock_late':   ('late',  'late',  'solid'),
    }
    label_for = {
        'earlyBlock_early': 'Early block, early in trial',
        'lateBlock_early':  'Late block, early in trial',
        'lateBlock_late':   'Late block, late in trial',
    }

    def get_probs(block, period):
        """n-weighted average of lickProb across the period's sliding windows"""
        starts = period_starts[period]
        subjs = list(pulse_lick_prob.keys())
        sums = np.zeros((len(subjs), len(bin_centres)))
        ns = np.zeros((len(subjs), len(bin_centres)))
        for t in starts:
            key = f'{block}Block_{t:.0f}-{t + time_win:.0f}s'
            for i, s in enumerate(subjs):
                if key not in pulse_lick_prob[s]:
                    continue
                p = pulse_lick_prob[s][key]['lickProb']
                n = pulse_lick_prob[s][key]['n']
                ok = ~np.isnan(p)
                sums[i, ok] += p[ok] * n[ok]
                ns[i, ok] += n[ok]
        probs = np.where(ns > 0, sums / np.maximum(ns, 1), np.nan)
        probs[:, ns.sum(axis=0) < min_n] = np.nan
        return probs

    cond_data = {name: get_probs(block, period)
                 for name, (block, period, _) in cond_specs.items()}

    # global baseline: per-subject P(lick) near tf_dev=0 in lateBlock_early
    gbl = cond_data['lateBlock_early']
    global_baseline = (np.nanmean(gbl[:, baseline_mask], axis=1, keepdims=True)
                       if gbl is not None else None)

    norm_titles = ['Raw', 'Per-block norm', 'Global norm (late baseline)']
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.07, row_titles=norm_titles)

    for name, (key, block, dash) in cond_specs.items():
        probs = cond_data[name]
        if probs is None:
            continue
        colour = _block_colour(block)
        colour_rgba = _block_rgba(block, 0.15)

        per_cond_baseline = np.nanmean(probs[:, baseline_mask],
                                       axis=1, keepdims=True)

        normed = {
            0: probs,
            1: probs - per_cond_baseline,
            2: probs - global_baseline if global_baseline is not None else None,
        }

        for row_idx, data in normed.items():
            if data is None:
                continue
            n_valid = np.sum(~np.isnan(data), axis=0).astype(float)
            mean = np.nanmean(data, axis=0)
            ci = 1.96 * np.nanstd(data, axis=0) / np.sqrt(
                np.where(n_valid > 0, n_valid, 1))
            show_legend = (row_idx == 0)

            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean + ci, mode='lines',
                line=dict(width=0), showlegend=False),
                row=row_idx + 1, col=1)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean - ci, mode='lines',
                line=dict(width=0), fill='tonexty',
                fillcolor=colour_rgba, showlegend=False),
                row=row_idx + 1, col=1)
            fig.add_trace(go.Scatter(
                x=bin_centres, y=mean, mode='lines',
                name=label_for[name],
                line=dict(color=colour, width=2, dash=dash),
                showlegend=show_legend),
                row=row_idx + 1, col=1)

        # zero line on the normalised rows
    for row in (2, 3):
        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                      row=row, col=1)

    fig.update_xaxes(range=xlims, row=3, col=1, title_text='TF deviation (oct)')
    for row in (1, 2, 3):
        fig.update_xaxes(range=xlims, row=row, col=1)
    fig.update_yaxes(title_text='P(lick)', row=1, col=1)
    fig.update_yaxes(title_text='delta P(lick)', row=2, col=1)
    fig.update_yaxes(title_text='delta P(lick)', row=3, col=1)
    fig.update_layout(template='plotly_white', width=600, height=750)
    return fig


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
                zlims[r] = [np.percentile(pooled, 0.5),
                            np.percentile(pooled, 99.5)]
            else:
                vmax = np.percentile(np.abs(pooled), 99.5)
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
                    # white -> dark red, the upper (positive) half of RdBu_r
                    cscale = [
                        [0.0, 'rgb(247,247,247)'],
                        [0.2, 'rgb(253,219,199)'],
                        [0.4, 'rgb(244,165,130)'],
                        [0.6, 'rgb(214,96,77)'],
                        [0.8, 'rgb(178,24,43)'],
                        [1.0, 'rgb(103,0,31)'],
                    ]
                    cmid = None
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


#%% pipeline entry points

def plot_all_behavioural(plot_dir):
    """load all cached behavioural extractions and save every plot"""
    from pathlib import Path
    from behaviour.extraction import load_behavioural
    from behaviour.two_pulse_analyses import (plot_two_pulse_interaction,
        plot_two_pulse_raw)
    from utils.figures import save_fig

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    psycho, chrono, _, _ = load_behavioural('psychometric')
    save_fig(plot_psychometric(psycho), str(plot_dir / 'psychometric'))
    save_fig(plot_psychometric(chrono), str(plot_dir / 'chronometric'))

    save_fig(plot_elta(load_behavioural('elta')), str(plot_dir / 'elta'))

    eltc = load_behavioural('eltc_aligned')
    save_fig(plot_eltc(eltc), str(plot_dir / 'eltc'))
    save_fig(plot_eltc_comparison(eltc), str(plot_dir / 'eltc_comparison'))

    save_fig(plot_el_hazard_rates(load_behavioural('hazard_rates')),
             str(plot_dir / 'hazard_rates'))

    pulse_lick = load_behavioural('pulse_lick_prob')
    for label, fig in plot_pulse_aligned_lick_prob(pulse_lick).items():
        save_fig(fig, str(plot_dir / f'pulse_lick_prob_{label}'))
    for label, fig in plot_pulse_lick_prob_2d(pulse_lick).items():
        save_fig(fig, str(plot_dir / f'pulse_lick_prob_2d_{label}'))

    two_pulse = load_behavioural('two_pulse_interaction')
    save_fig(plot_two_pulse_interaction(two_pulse),
             str(plot_dir / 'two_pulse_interaction'))
    save_fig(plot_two_pulse_raw(two_pulse), str(plot_dir / 'two_pulse_raw'))


def plot_all_quantifications(plot_dir):
    """compute (or load cached) behavioural stats and save every quant plot"""
    from pathlib import Path
    from behaviour.extraction import load_behavioural
    from behaviour.quantification import run_all_quantifications
    from utils.figures import save_fig

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    _, _, n_hits, n_trials = load_behavioural('psychometric')
    stats = run_all_quantifications(overwrite=False)

    psycho_params, _, _, _ = stats['psychometric']
    save_fig(plot_psychometric_fits(psycho_params, n_hits, n_trials),
             str(plot_dir / 'psychometric_fits'))
    save_fig(plot_lts_quant(stats['lts']), str(plot_dir / 'lts_quant'))
    save_fig(plot_lts_pcs(stats['lts']), str(plot_dir / 'lts_pcs'))
    save_fig(plot_hazard_rate_stats(stats['hazard_rates']),
             str(plot_dir / 'hazard_rate_stats'))
    save_fig(plot_pulse_lick_fits(stats['pulse_lick_prob']),
             str(plot_dir / 'pulse_lick_fits'))
    save_fig(plot_integration_time(stats['integration_time']),
             str(plot_dir / 'integration_time'))
