"""
plotly visualisations for the behaviour RNN:
- training curves per mouse
- real-vs-RNN mirror plots for the three mouse behavioural observables
  (FA hazard, pulse-aligned lick prob, lick-triggered TF). these mirror the
  mouse-only plots in behaviour.plotting (2-row mean+delta, CI ribbons,
  change-window shading) with the RNN overlaid as a dotted line.
"""
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import ANALYSIS_OPTIONS, PLOT_OPTIONS
from behaviour.plotting import _block_colour, _block_rgba
from behaviour.extraction import load_behavioural
from behaviour_rnn.train import load_rnn_results
from utils.figures import save_fig


#%% training curves

def plot_training_curves(results_by_subj):
    """train/val BCE per epoch, one panel per mouse"""
    subjs = list(results_by_subj.keys())
    n = len(subjs)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subjs,
                        shared_yaxes=False, vertical_spacing=0.12,
                        horizontal_spacing=0.08)

    for i, subj in enumerate(subjs):
        r, c = i // n_cols + 1, i % n_cols + 1
        h = results_by_subj[subj]['history']
        n_h = results_by_subj[subj].get('n_hidden', '?')
        epochs = np.arange(len(h['train']))
        fig.add_trace(go.Scatter(x=epochs, y=h['train'], mode='lines',
            line=dict(color='steelblue'), name='train',
            showlegend=(i == 0)), row=r, col=c)
        fig.add_trace(go.Scatter(x=epochs, y=h['val'], mode='lines',
            line=dict(color='firebrick'), name='val',
            showlegend=(i == 0)), row=r, col=c)
        fig.layout.annotations[i].text = f'{subj} (n_h={n_h})'

    fig.update_xaxes(title_text='epoch', row=n_rows)
    fig.update_yaxes(title_text='BCE', col=1)
    fig.update_layout(template='plotly_white', height=250 * n_rows, width=900)
    return fig


#%% real vs RNN: hazard rate

def _stack_block(d, block_key, subj_keys):
    return np.stack([d[s][block_key] for s in subj_keys]).astype(float)


def plot_hazard_real_vs_rnn(real, sim, config=ANALYSIS_OPTIONS):
    """FA hazard mouse vs RNN, mirroring plot_el_hazard_rates: rate + delta rows,
    1.96 SEM ribbon around the mouse mean, dotted RNN overlay (mean only),
    change-window shading on both rows."""
    block_keys = {'early': 'earlyBlock', 'late': 'lateBlock'}
    n_keys = {'early': 'early_n', 'late': 'late_n'}
    min_n = config.get('min_hazard_samples', 100)
    change_wins = config.get('change_wins', {})

    subj_keys = [s for s in real if s in sim]
    if not subj_keys:
        return go.Figure()
    bin_centres = real[subj_keys[0]]['binCentres']

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)

    for block in ['early', 'late']:
        if block not in change_wins:
            continue
        w = change_wins[block]
        for xref, yref in [('x', 'y'), ('x2', 'y2')]:
            fig.add_shape(type='rect', x0=w[0], x1=w[1], y0=0, y1=1,
                          xref=xref, yref=f'{yref} domain',
                          fillcolor=_block_rgba(block, 0.12),
                          line_width=0, layer='below')

    real_block, sim_block = {}, {}
    for block in ['early', 'late']:
        colour = _block_colour(block)
        rgba = _block_rgba(block, 0.15)
        bkey = block_keys[block]
        nkey = n_keys[block]

        all_real = np.stack([real[s][bkey] for s in subj_keys]).astype(float)
        all_sim  = np.stack([sim[s][bkey]  for s in subj_keys]).astype(float)
        all_n    = np.stack([real[s][nkey] for s in subj_keys])
        all_real[all_n < min_n] = np.nan
        all_sim[all_n < min_n]  = np.nan
        real_block[block], sim_block[block] = all_real, all_sim

        n_valid = np.sum(~np.isnan(all_real), axis=0).astype(float)
        mean_r = np.nanmean(all_real, axis=0)
        ci_r = 1.96 * np.nanstd(all_real, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))
        mean_s = np.nanmean(all_sim, axis=0)

        fig.add_trace(go.Scatter(x=bin_centres, y=mean_r + ci_r, mode='lines',
            line=dict(width=0), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=bin_centres, y=mean_r - ci_r, mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor=rgba,
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=bin_centres, y=mean_r, mode='lines',
            name=f'{block.capitalize()} block',
            line=dict(color=colour, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bin_centres, y=mean_s, mode='lines',
            name=f'{block.capitalize()} (RNN)',
            line=dict(color=colour, width=2, dash='dot')), row=1, col=1)

    delta_r = real_block['early'] - real_block['late']
    delta_s = sim_block['early']  - sim_block['late']
    n_valid = np.sum(~np.isnan(delta_r), axis=0).astype(float)
    mean_dr = np.nanmean(delta_r, axis=0)
    ci_dr = 1.96 * np.nanstd(delta_r, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))
    mean_ds = np.nanmean(delta_s, axis=0)

    fig.add_trace(go.Scatter(x=bin_centres, y=mean_dr + ci_dr, mode='lines',
        line=dict(width=0), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=bin_centres, y=mean_dr - ci_dr, mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor='rgba(128,128,128,0.15)',
        showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=bin_centres, y=mean_dr, mode='lines',
        name='Early - Late', line=dict(color='grey', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=bin_centres, y=mean_ds, mode='lines',
        name='Early - Late (RNN)',
        line=dict(color='grey', width=2, dash='dot')), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                  row=2, col=1)

    fig.update_layout(template='plotly_white', width=600, height=500)
    fig.update_xaxes(title_text='Time in trial (s)', row=2, col=1)
    fig.update_yaxes(title_text='FA hazard rate', row=1, col=1)
    fig.update_yaxes(title_text='Delta hazard (Early - Late)', row=2, col=1)
    return fig


#%% real vs RNN: pulse-aligned lick probability

def plot_pulse_lick_real_vs_rnn(real, sim, config=ANALYSIS_OPTIONS):
    """one figure per time window. mirrors plot_pulse_aligned_lick_prob:
    row 1 = mouse early/late lick prob (CI ribbon) + RNN dotted overlay;
    row 2 = paired Early-Late delta (mouse mean + CI, RNN dotted)."""
    subj_keys = [s for s in real if s in sim]
    if not subj_keys:
        return {}
    bin_centres = np.array(real[subj_keys[0]]['binCentres'])
    time_starts = real[subj_keys[0]]['time_starts']
    time_win    = real[subj_keys[0]]['time_win']
    min_n = config.get('min_pulse_samples', 500)
    xlims = [-0.6, 0.6]

    def _gather(d, cond):
        arr = []
        n   = []
        for s in subj_keys:
            if cond not in d[s]:
                arr.append(None)
                continue
            arr.append(np.asarray(d[s][cond]['lickProb'], dtype=float))
            n.append(np.asarray(d[s][cond].get('n', np.zeros_like(arr[-1]))))
        if any(a is None for a in arr):
            return None, None
        return np.stack(arr), np.stack(n) if n else None

    figs = {}
    for t_start in time_starts:
        t_end = t_start + time_win
        label = f'{t_start:.0f}-{t_end:.0f}s'
        early_key = f'earlyBlock_{label}'
        late_key  = f'lateBlock_{label}'

        real_e, n_e = _gather(real, early_key)
        real_l, n_l = _gather(real, late_key)
        sim_e, _    = _gather(sim,  early_key)
        sim_l, _    = _gather(sim,  late_key)
        if any(x is None for x in (real_e, real_l, sim_e, sim_l)):
            continue
        if n_e is not None:
            real_e[:, n_e.sum(axis=0) < min_n] = np.nan
        if n_l is not None:
            real_l[:, n_l.sum(axis=0) < min_n] = np.nan

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.6, 0.4], vertical_spacing=0.08)

        for block, real_arr, sim_arr in (('early', real_e, sim_e),
                                         ('late',  real_l, sim_l)):
            colour = _block_colour(block)
            rgba   = _block_rgba(block, 0.15)
            n_valid = np.sum(~np.isnan(real_arr), axis=0).astype(float)
            mean_r  = np.nanmean(real_arr, axis=0)
            ci_r    = 1.96 * np.nanstd(real_arr, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))
            mean_s  = np.nanmean(sim_arr, axis=0)

            fig.add_trace(go.Scatter(x=bin_centres, y=mean_r + ci_r, mode='lines',
                line=dict(width=0), showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=bin_centres, y=mean_r - ci_r, mode='lines',
                line=dict(width=0), fill='tonexty', fillcolor=rgba,
                showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=bin_centres, y=mean_r, mode='lines',
                name=f'{block.capitalize()} block',
                line=dict(color=colour, width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=bin_centres, y=mean_s, mode='lines',
                name=f'{block.capitalize()} (RNN)',
                line=dict(color=colour, width=2, dash='dot')), row=1, col=1)

        delta_r = real_e - real_l
        delta_s = sim_e  - sim_l
        n_valid = np.sum(~np.isnan(delta_r), axis=0).astype(float)
        mean_dr = np.nanmean(delta_r, axis=0)
        ci_dr   = 1.96 * np.nanstd(delta_r, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))
        mean_ds = np.nanmean(delta_s, axis=0)

        fig.add_trace(go.Scatter(x=bin_centres, y=mean_dr + ci_dr, mode='lines',
            line=dict(width=0), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=bin_centres, y=mean_dr - ci_dr, mode='lines',
            line=dict(width=0), fill='tonexty', fillcolor='rgba(128,128,128,0.15)',
            showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=bin_centres, y=mean_dr, mode='lines',
            name='Early - Late', line=dict(color='grey', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=bin_centres, y=mean_ds, mode='lines',
            name='Early - Late (RNN)',
            line=dict(color='grey', width=2, dash='dot')), row=2, col=1)
        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                      row=2, col=1)

        fig.update_xaxes(range=xlims, row=1, col=1)
        fig.update_xaxes(range=xlims, title_text='Delta TF (oct)', row=2, col=1)
        fig.update_yaxes(title_text='P(lick in window)', row=1, col=1)
        fig.update_yaxes(title_text='Delta P(lick) (Early - Late)', row=2, col=1)
        fig.update_layout(template='plotly_white', width=600, height=500,
                          title_text=label)
        figs[label] = fig

    return figs


#%% real vs RNN: lick-triggered TF kernel

def plot_kernel_real_vs_rnn(real_elta, sim_kernel, config=ANALYSIS_OPTIONS):
    """mirrors plot_elta. row 1 = mouse 3-condition kernels (CI ribbons) with
    RNN dotted overlay; row 2 = paired Early-Late delta on early-trial licks
    (mouse mean + CI, RNN dotted)."""
    n_pre = config.get('n_pre_lick_samples', 40)
    sample_rate = config.get('tf_sample_rate', 20)
    t = np.linspace(-n_pre / sample_rate, 0, n_pre)

    line_specs = {
        'earlyBlock_early': {'block': 'early', 'dash': 'solid',
                             'label': 'Early block, early lick'},
        'lateBlock_early':  {'block': 'late',  'dash': 'dash',
                             'label': 'Late block, early lick'},
        'lateBlock_late':   {'block': 'late',  'dash': 'solid',
                             'label': 'Late block, late lick'},
    }

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)

    for cond, spec in line_specs.items():
        if cond not in real_elta:
            continue
        scale = 0.65 if spec['dash'] == 'dash' else 1.0
        colour = _block_colour(spec['block'], scale)
        rgba   = _block_rgba(spec['block'], 0.2, scale)

        mean_r = real_elta[cond]['mean']
        sem_r  = real_elta[cond]['sem']
        fig.add_trace(go.Scatter(
            x=t, y=mean_r, mode='lines', name=spec['label'],
            line=dict(color=colour, dash=spec['dash'], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(mean_r + sem_r) + list(mean_r - sem_r)[::-1],
            fill='toself', fillcolor=rgba, line=dict(width=0),
            mode='none', showlegend=False), row=1, col=1)

        sim_subjs = [s for s in sim_kernel if cond in sim_kernel[s]]
        if sim_subjs:
            sim_arr = np.stack([sim_kernel[s][cond] for s in sim_subjs]).astype(float)
            mean_s = np.nanmean(sim_arr, axis=0)
            fig.add_trace(go.Scatter(
                x=t, y=mean_s, mode='lines', name=f"{spec['label']} (RNN)",
                line=dict(color=colour, width=2, dash='dot')), row=1, col=1)

    # delta: early-trial licks, early block minus late block (paired across mice).
    # only available when called with cohort elta (subj_means present); single-subj
    # path from plot_subject_mirrors passes only 'mean'/'sem' so skip the delta there.
    has_cohort = ('earlyBlock_early' in real_elta and 'lateBlock_early' in real_elta
                  and 'subj_means' in real_elta['earlyBlock_early']
                  and 'subj_means' in real_elta['lateBlock_early'])
    if has_cohort:
        e, l = real_elta['earlyBlock_early'], real_elta['lateBlock_early']
        e_subjs = e.get('subjs')
        l_subjs = l.get('subjs')
        if e_subjs is not None and l_subjs is not None:
            common = [s for s in e_subjs if s in l_subjs]
            e_idx = [e_subjs.index(s) for s in common]
            l_idx = [l_subjs.index(s) for s in common]
            delta = e['subj_means'][e_idx] - l['subj_means'][l_idx]
        else:
            delta = e['subj_means'] - l['subj_means']

        n_valid = np.sum(~np.isnan(delta), axis=0).astype(float)
        mean_d = np.nanmean(delta, axis=0)
        sem_d = np.nanstd(delta, axis=0) / np.sqrt(np.where(n_valid > 0, n_valid, 1))

        fig.add_trace(go.Scatter(
            x=t, y=mean_d, mode='lines',
            name='Early - Late (early-trial licks)',
            line=dict(color='grey', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(mean_d + sem_d) + list(mean_d - sem_d)[::-1],
            fill='toself', fillcolor='rgba(128,128,128,0.2)',
            line=dict(width=0), mode='none', showlegend=False), row=2, col=1)

        common_sim = [s for s in sim_kernel
                      if 'earlyBlock_early' in sim_kernel[s]
                      and 'lateBlock_early' in sim_kernel[s]]
        if common_sim:
            d_sim = np.stack([sim_kernel[s]['earlyBlock_early']
                              - sim_kernel[s]['lateBlock_early']
                              for s in common_sim]).astype(float)
            mean_ds = np.nanmean(d_sim, axis=0)
            fig.add_trace(go.Scatter(
                x=t, y=mean_ds, mode='lines',
                name='Early - Late (RNN)',
                line=dict(color='grey', width=2, dash='dot')), row=2, col=1)

        fig.add_hline(y=0, line=dict(color='black', width=1, dash='dot'),
                      row=2, col=1)

    fig.update_xaxes(title_text='Time relative to lick (s)', row=2, col=1)
    fig.update_yaxes(title_text='Baseline stimulus (octaves)', row=1, col=1)
    fig.update_yaxes(title_text='Delta stimulus (octaves)', row=2, col=1)
    fig.update_layout(template='plotly_white', width=700, height=550)
    return fig


#%% outcome distribution (Hit / FA / Miss)

def plot_outcome_dist_single(dist):
    """one mouse, three panels (Hit / FA / Miss): mouse vs RNN proportion by block"""
    outcomes = ['hit', 'fa', 'miss']
    fig = make_subplots(rows=1, cols=3, subplot_titles=[o.upper() for o in outcomes],
                        shared_yaxes=True, horizontal_spacing=0.06)
    for c, oc in enumerate(outcomes, start=1):
        for block_name, data in dist.items():
            colour = _block_colour(block_name)
            fig.add_trace(go.Scatter(
                x=['mouse', 'RNN'],
                y=[data['mouse'][oc], data['model'][oc]],
                mode='lines+markers',
                line=dict(color=colour, width=2.5),
                marker=dict(size=10),
                name=block_name, legendgroup=block_name,
                showlegend=(c == 1)),
                row=1, col=c)
    fig.update_yaxes(range=[0, 1], title_text='proportion', col=1)
    fig.update_layout(template='plotly_white', width=720, height=320)
    return fig


def plot_outcome_dist_cohort(dist_by_subj):
    """three panels (Hit / FA / Miss): thin per-mouse mouse-vs-RNN lines + group mean by block"""
    outcomes = ['hit', 'fa', 'miss']
    fig = make_subplots(rows=1, cols=3, subplot_titles=[o.upper() for o in outcomes],
                        shared_yaxes=True, horizontal_spacing=0.06)
    for c, oc in enumerate(outcomes, start=1):
        for block_name in ['early', 'late']:
            colour = _block_colour(block_name)
            mice = [s for s in dist_by_subj if block_name in dist_by_subj[s]]
            if not mice:
                continue
            mouse_p = np.array([dist_by_subj[s][block_name]['mouse'][oc] for s in mice])
            model_p = np.array([dist_by_subj[s][block_name]['model'][oc] for s in mice])
            for i in range(len(mice)):
                fig.add_trace(go.Scatter(
                    x=['mouse', 'RNN'], y=[mouse_p[i], model_p[i]],
                    mode='lines+markers',
                    line=dict(color=colour, width=1), opacity=0.35,
                    marker=dict(size=4),
                    showlegend=False, hoverinfo='skip'),
                    row=1, col=c)
            fig.add_trace(go.Scatter(
                x=['mouse', 'RNN'],
                y=[np.nanmean(mouse_p), np.nanmean(model_p)],
                mode='lines+markers',
                line=dict(color=colour, width=3),
                marker=dict(size=10),
                name=block_name, legendgroup=block_name,
                showlegend=(c == 1)),
                row=1, col=c)
    fig.update_yaxes(range=[0, 1], title_text='proportion', col=1)
    fig.update_layout(template='plotly_white', width=900, height=380)
    return fig


#%% example-trial panels

def plot_example_trials(model, df_subj, pos_weight, config=ANALYSIS_OPTIONS):
    """heatmaps of all trials per (block, outcome), sorted by lick time.
    left col = stimulus TF; right col = RNN P(lick). lick / change times
    overlaid as markers. mirrors lick_pred.analysis.plot_session_heatmap."""
    from behaviour_rnn.simulate_behaviour import predict_for_df
    p_lick, tf_in, meta = predict_for_df(model, df_subj, pos_weight)
    df = meta['df']
    dt = meta['dt']
    max_t = p_lick.shape[1]
    t_grid = np.arange(max_t) * dt

    stim_t = df['stimT'].to_numpy(dtype=float)
    rt_rt  = df['rt_RT'].to_numpy(dtype=float)
    rt_fa  = df['rt_FA'].to_numpy(dtype=float)

    pl = np.where(np.isnan(p_lick), 0.0, p_lick)

    blocks_def = [('early', 1.0), ('late', -1.0)]
    outcomes   = [('hit', meta['is_hit']),
                  ('fa',  meta['is_fa']),
                  ('miss', meta['is_miss'])]

    cells = []  # (label, sorted trial indices, lick_t (or None), stim_t)
    for block_name, block_val in blocks_def:
        for outcome_name, outcome_mask in outcomes:
            sel = np.where((meta['blocks'] == block_val) & outcome_mask)[0]
            if outcome_name == 'hit':
                lick = stim_t[sel] + rt_rt[sel]
            elif outcome_name == 'fa':
                lick = rt_fa[sel]
            else:
                lick = stim_t[sel]
            order = np.argsort(lick) if len(sel) else np.array([], dtype=int)
            cells.append((f'{block_name} {outcome_name}',
                          sel[order], lick[order] if len(sel) else None,
                          stim_t[sel[order]] if len(sel) else None))

    n_rows = len(cells)
    fig = make_subplots(rows=n_rows, cols=2,
                        row_titles=[c[0] for c in cells],
                        column_titles=['Stimulus (TF, oct)', 'RNN P(lick)'],
                        shared_xaxes=True, shared_yaxes=False,
                        horizontal_spacing=0.08, vertical_spacing=0.015)

    for r, (label, idx, lick, stm) in enumerate(cells, start=1):
        if len(idx) == 0:
            continue
        y_pos = np.arange(len(idx))

        fig.add_trace(go.Heatmap(
            z=tf_in[idx], x=t_grid, y=y_pos, coloraxis='coloraxis',
            hoverinfo='skip'), row=r, col=1)
        fig.add_trace(go.Heatmap(
            z=pl[idx], x=t_grid, y=y_pos, coloraxis='coloraxis2',
            hoverinfo='skip'), row=r, col=2)

        # change-onset (black) on both panels
        if stm is not None and np.isfinite(stm).any():
            for c_i in (1, 2):
                fig.add_trace(go.Scatter(
                    x=stm, y=y_pos, mode='lines',
                    line=dict(color='black', width=1, dash='dot'),
                    showlegend=False, hoverinfo='skip'),
                    row=r, col=c_i)

        # lick time (green hit / red FA, no marker for miss)
        outcome = label.split()[-1]
        if outcome in ('hit', 'fa') and np.isfinite(lick).any():
            colour = 'seagreen' if outcome == 'hit' else 'crimson'
            for c_i in (1, 2):
                fig.add_trace(go.Scatter(
                    x=lick, y=y_pos, mode='lines',
                    line=dict(color=colour, width=1.5),
                    showlegend=False, hoverinfo='skip'),
                    row=r, col=c_i)

        fig.update_yaxes(title_text=f'trials (n={len(idx)})',
                         row=r, col=1, autorange='reversed')
        fig.update_yaxes(showticklabels=False, row=r, col=2, autorange='reversed')

    fig.update_xaxes(title_text='Time in trial (s)', row=n_rows)
    fig.update_layout(
        template='plotly_white',
        height=140 * n_rows, width=950,
        coloraxis  = dict(colorscale='RdBu_r', cmin=-1.5, cmax=1.5, cmid=0,
                          colorbar=dict(title='TF', x=0.46, len=0.4, y=0.5)),
        coloraxis2 = dict(colorscale='Reds', cmin=0, cmax=1,
                          colorbar=dict(title='P(lick)', x=1.02, len=0.4, y=0.5)),
        showlegend=False,
    )
    return fig


#%% per-trial PDF (mirrors lick_pred.plot_all_lick_trials)

def plot_all_rnn_trials(model, df_subj, pos_weight, save_path,
                        config=ANALYSIS_OPTIONS):
    """multipage PDF, one page per trial.
    row 1: TF trace + change-onset (black dotted) + mouse lick (green hit / red FA).
    row 2: training target + RNN P(lick), same markers.
    title: block, mouse outcome, model outcome (argmax of analytic P_fa/P_hit/P_miss).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.backends.backend_pdf import PdfPages
    from behaviour_rnn.train import build_tensors
    from behaviour_rnn.simulate_behaviour import predict_p_lick, _apply_motor_delay

    inputs, target, mask, meta = build_tensors(df_subj)
    p_lick = predict_p_lick(model, inputs, mask, pos_weight)

    iti = meta.get('iti_bins', 0)
    if iti > 0:
        p_lick    = p_lick[:, iti:]
        target_np = target.numpy()[:, iti:]
        tf_in     = inputs[:, iti:, 0].numpy()
    else:
        target_np = target.numpy()
        tf_in     = inputs[:, :, 0].numpy()
    p_lick = _apply_motor_delay(p_lick, meta['rt_samples'], meta['dt'])

    df    = meta['df']
    dt    = meta['dt']
    n, T  = p_lick.shape
    t_grid = np.arange(T) * dt

    # per-trial model outcome label (argmax of analytic P_fa, P_hit_cond, P_miss_cond)
    rw_bins = int(round(config.get('response_window', 2.15) / dt))
    p_safe  = np.where(np.isnan(p_lick), 0.0, p_lick)
    log_s   = np.log(np.clip(1.0 - p_safe, 1e-9, 1.0))
    cumlog  = np.concatenate([np.zeros((n, 1)), np.cumsum(log_s, axis=1)[:, :-1]], axis=1)
    p_first = p_safe * np.exp(cumlog)

    stim_t     = df['stimT'].to_numpy(dtype=float)
    has_change = np.isfinite(stim_t)
    change_bin = np.where(has_change,
                          np.clip(np.ceil(stim_t / dt).astype(int) - 1, 0, T - 1),
                          T)

    model_outcome = np.empty(n, dtype=object)
    for i in range(n):
        cb = int(change_bin[i])
        p_fa = float(p_first[i, :cb].sum()) if cb > 0 else 0.0
        if has_change[i] and cb < T:
            saw = float(np.exp(cumlog[i, cb]))
            p_hit_u = float(p_first[i, cb:min(cb + rw_bins, T)].sum())
            p_hit = p_hit_u / saw if saw > 1e-9 else 0.0
            p_miss = max(0.0, 1.0 - p_hit)
            joint = np.array([p_fa, (1 - p_fa) * p_hit, (1 - p_fa) * p_miss])
        else:
            joint = np.array([p_fa, 0.0, 1.0 - p_fa])
        model_outcome[i] = ['fa', 'hit', 'miss'][int(joint.argmax())]

    mouse_outcome = np.where(df['IsHit'], 'hit',
                     np.where(df['IsFA'], 'fa',
                     np.where(df['IsMiss'], 'miss', 'other')))
    blocks_str = np.where(meta['blocks'] == 1.0, 'early', 'late')
    rt_rt = df['rt_RT'].to_numpy(dtype=float)
    rt_fa = df['rt_FA'].to_numpy(dtype=float)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(save_path) as pdf:
        for i in range(n):
            fig, (ax_tf, ax_p) = plt.subplots(
                2, 1, figsize=(8, 5), sharex=True,
                gridspec_kw={'height_ratios': [1, 1.4]})

            ax_tf.plot(t_grid, tf_in[i], color='k', linewidth=1)
            ax_tf.axhline(0, color='grey', linewidth=0.3)
            ax_tf.set_ylabel('TF (oct)')

            if has_change[i]:
                for ax in (ax_tf, ax_p):
                    ax.axvline(stim_t[i], color='black', linestyle=':', linewidth=1)

            if mouse_outcome[i] == 'hit' and np.isfinite(rt_rt[i]):
                lick_t, colour = stim_t[i] + rt_rt[i], 'seagreen'
            elif mouse_outcome[i] == 'fa' and np.isfinite(rt_fa[i]):
                lick_t, colour = rt_fa[i], 'crimson'
            else:
                lick_t = None
            if lick_t is not None:
                for ax in (ax_tf, ax_p):
                    ax.axvline(lick_t, color=colour, linewidth=1)

            ax_p.plot(t_grid, target_np[i], color='grey', alpha=0.7, label='target')
            ax_p.plot(t_grid, p_lick[i],    color='tab:red',          label='RNN P(lick)')
            ax_p.set_ylim(-0.05, 1.05)
            ax_p.set_ylabel('P(lick)')
            ax_p.set_xlabel('Time in trial (s)')
            ax_p.legend(loc='upper right', fontsize=8)

            ax_tf.set_title(
                f'trial {i + 1} | {blocks_str[i]} block | '
                f'mouse: {mouse_outcome[i]} | model: {model_outcome[i]}')
            sns.despine(ax=ax_tf)
            sns.despine(ax=ax_p)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'saved {n} trial plots to {save_path}')


#%% single-subject plots (used during n_hidden sweep)

def plot_subject_mirrors(subj, sim, plot_dir, real_cached=None,
                         config=ANALYSIS_OPTIONS):
    """write hazard / pulse / kernel real-vs-RNN plots for one mouse.
    sim has keys 'hazard', 'pulse', 'kernel', each a single-subject dict
    (i.e. what simulate_all returns)."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if real_cached is None:
        real_cached = dict(
            hazard = load_behavioural('hazard_rates'),
            pulse  = load_behavioural('pulse_lick_prob'),
            elta   = load_behavioural('elta'),
        )

    real_h = {subj: real_cached['hazard'][subj]} if subj in real_cached['hazard'] else {}
    sim_h  = {subj: sim['hazard']}
    if real_h:
        save_fig(plot_hazard_real_vs_rnn(real_h, sim_h, config),
                 str(plot_dir / 'hazard_rates'))

    # pulse_lick_prob plots disabled (slow)
    # real_p = {subj: real_cached['pulse'][subj]} if subj in real_cached['pulse'] else {}
    # sim_p  = {subj: sim['pulse']}
    # if real_p:
    #     for cond, fig in plot_pulse_lick_real_vs_rnn(real_p, sim_p, config).items():
    #         save_fig(fig, str(plot_dir / f'pulse_lick_prob_{cond}'))

    # elta is cohort-aggregated (mean/sem across subjs); rebuild for this subj only
    real_elta_subj = {}
    for cond, data in real_cached['elta'].items():
        subjs = list(data.get('subjs', []))
        if subj in subjs:
            m = data['subj_means'][subjs.index(subj)]
            real_elta_subj[cond] = {'mean': m, 'sem': np.zeros_like(m)}
    if real_elta_subj:
        save_fig(plot_kernel_real_vs_rnn(real_elta_subj,
                                         {subj: sim['kernel']}, config),
                 str(plot_dir / 'elta'))


def load_real_observables():
    """convenience loader for the three real observables used in mirror plots"""
    return dict(
        hazard = load_behavioural('hazard_rates'),
        pulse  = load_behavioural('pulse_lick_prob'),
        elta   = load_behavioural('elta'),
    )


#%% pipeline entry point

def comparative_plots(plot_dir):
    """make all RNN diagnostic plots: training curves + real-vs-RNN mirrors"""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rnn, sim = load_rnn_results()

    save_fig(plot_training_curves(rnn), str(plot_dir / 'training_curves'))
    save_fig(plot_hazard_real_vs_rnn(load_behavioural('hazard_rates'), sim['hazard']),
             str(plot_dir / 'hazard_rates'))
    # pulse_lick_prob plots disabled (slow)
    # for cond, fig in plot_pulse_lick_real_vs_rnn(
    #         load_behavioural('pulse_lick_prob'), sim['pulse']).items():
    #     save_fig(fig, str(plot_dir / f'pulse_lick_prob_{cond}'))
    save_fig(plot_kernel_real_vs_rnn(load_behavioural('elta'), sim['kernel']),
             str(plot_dir / 'elta'))
    save_fig(plot_outcome_dist_cohort(sim['outcome']),
             str(plot_dir / 'outcome'))
