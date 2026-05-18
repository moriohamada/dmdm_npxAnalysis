"""
plotly visualisations for the behaviour RNN:
- training curves per mouse
- real-vs-RNN mirror plots for the three mouse behavioural observables
  (FA hazard, pulse-aligned lick prob, lick-triggered TF)
"""
from pathlib import Path
import numpy as np
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
    """overlay mouse and RNN FA hazard, per block, with per-mouse lines + group mean"""
    subj_keys = [s for s in real if s in sim]
    bin_centres = real[subj_keys[0]]['binCentres']
    change_wins = config.get('change_wins', {})

    fig = go.Figure()

    for block, key in [('early', 'earlyBlock'), ('late', 'lateBlock')]:
        if block in change_wins:
            w = change_wins[block]
            fig.add_vrect(x0=w[0], x1=w[1], fillcolor=_block_rgba(block, 0.10),
                          line_width=0, layer='below')

        real_arr = _stack_block(real, key, subj_keys)
        sim_arr  = _stack_block(sim,  key, subj_keys)

        colour = _block_colour(block)
        # per-mouse thin lines
        for i in range(real_arr.shape[0]):
            fig.add_trace(go.Scatter(
                x=bin_centres, y=real_arr[i], mode='lines',
                line=dict(color=colour, width=1), opacity=0.25,
                showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(
                x=bin_centres, y=sim_arr[i], mode='lines',
                line=dict(color=colour, width=1, dash='dot'), opacity=0.25,
                showlegend=False, hoverinfo='skip'))
        # group means
        fig.add_trace(go.Scatter(
            x=bin_centres, y=np.nanmean(real_arr, axis=0),
            mode='lines', name=f'{block} mouse',
            line=dict(color=colour, width=2.5)))
        fig.add_trace(go.Scatter(
            x=bin_centres, y=np.nanmean(sim_arr, axis=0),
            mode='lines', name=f'{block} RNN',
            line=dict(color=colour, width=2.5, dash='dot')))

    fig.update_xaxes(title_text='time in trial (s)')
    fig.update_yaxes(title_text='FA hazard')
    fig.update_layout(template='plotly_white', width=700, height=450)
    return fig


#%% real vs RNN: pulse-aligned lick probability

def plot_pulse_lick_real_vs_rnn(real, sim, config=ANALYSIS_OPTIONS):
    """one figure per (block, time window): real vs RNN lick prob over TF magnitude"""
    subj_keys = [s for s in real if s in sim]
    bin_centres = np.array(real[subj_keys[0]]['binCentres'])
    figs = {}

    # collect common condition names from real
    cond_names = [k for k in real[subj_keys[0]]
                  if k not in ('binCentres', 'time_starts', 'time_win')]

    for cond in cond_names:
        if not all(cond in real[s] and cond in sim[s] for s in subj_keys):
            continue
        block = 'early' if cond.startswith('early') else 'late'
        colour = _block_colour(block)

        real_arr = np.stack([real[s][cond]['lickProb'] for s in subj_keys]).astype(float)
        sim_arr  = np.stack([sim[s][cond]['lickProb']  for s in subj_keys]).astype(float)

        fig = go.Figure()
        for i in range(real_arr.shape[0]):
            fig.add_trace(go.Scatter(
                x=bin_centres, y=real_arr[i], mode='lines',
                line=dict(color=colour, width=1), opacity=0.25,
                showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(
                x=bin_centres, y=sim_arr[i], mode='lines',
                line=dict(color=colour, width=1, dash='dot'), opacity=0.25,
                showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter(
            x=bin_centres, y=np.nanmean(real_arr, axis=0), mode='lines',
            name='mouse', line=dict(color=colour, width=2.5)))
        fig.add_trace(go.Scatter(
            x=bin_centres, y=np.nanmean(sim_arr, axis=0), mode='lines',
            name='RNN', line=dict(color=colour, width=2.5, dash='dot')))

        fig.update_xaxes(title_text='preceding TF (octaves)')
        fig.update_yaxes(title_text='P(lick in window)')
        fig.update_layout(template='plotly_white', width=600, height=400,
                          title_text=cond)
        figs[cond] = fig

    return figs


#%% real vs RNN: lick-triggered TF kernel

def plot_kernel_real_vs_rnn(real_elta, sim_kernel, config=ANALYSIS_OPTIONS):
    """overlay mouse and RNN lick-triggered TF for each condition"""
    n_pre = config.get('n_pre_lick_samples', 40)
    sample_rate = config.get('tf_sample_rate', 20)
    t = np.linspace(-n_pre / sample_rate, 0, n_pre)

    line_specs = {
        'earlyBlock_early': {'block': 'early', 'dash': 'solid',  'label': 'early block, early lick'},
        'lateBlock_early':  {'block': 'late',  'dash': 'dash',   'label': 'late block, early lick'},
        'lateBlock_late':   {'block': 'late',  'dash': 'solid',  'label': 'late block, late lick'},
    }

    fig = go.Figure()
    for cond, spec in line_specs.items():
        if cond not in real_elta:
            continue
        scale = 0.65 if spec['dash'] == 'dash' else 1.0
        colour = _block_colour(spec['block'], scale)

        # real
        mean_r = real_elta[cond]['mean']
        sem_r  = real_elta[cond]['sem']
        fig.add_trace(go.Scatter(
            x=t, y=mean_r, mode='lines', name=f"{spec['label']} (mouse)",
            line=dict(color=colour, width=2.5, dash=spec['dash'])))
        fig.add_trace(go.Scatter(
            x=list(t) + list(t)[::-1],
            y=list(mean_r + sem_r) + list(mean_r - sem_r)[::-1],
            fill='toself', fillcolor=_block_rgba(spec['block'], 0.18, scale),
            line=dict(width=0), mode='none', showlegend=False))

        # RNN: aggregate across mice
        sim_subjs = [s for s in sim_kernel if cond in sim_kernel[s]]
        if not sim_subjs:
            continue
        sim_arr = np.stack([sim_kernel[s][cond] for s in sim_subjs]).astype(float)
        mean_s = np.nanmean(sim_arr, axis=0)
        fig.add_trace(go.Scatter(
            x=t, y=mean_s, mode='lines', name=f"{spec['label']} (RNN)",
            line=dict(color=colour, width=2.5, dash='dot')))

    fig.update_xaxes(title_text='time relative to lick (s)')
    fig.update_yaxes(title_text='baseline TF (octaves)')
    fig.update_layout(template='plotly_white', width=700, height=450)
    return fig


#%% pipeline entry point

def comparative_plots(plot_dir):
    """make all RNN diagnostic plots: training curves + real-vs-RNN mirrors"""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rnn, sim = load_rnn_results()

    save_fig(plot_training_curves(rnn), str(plot_dir / 'training_curves'))
    save_fig(plot_hazard_real_vs_rnn(load_behavioural('hazard_rates'), sim['hazard']),
             str(plot_dir / 'hazard_real_vs_rnn'))
    for cond, fig in plot_pulse_lick_real_vs_rnn(
            load_behavioural('pulse_lick_prob'), sim['pulse']).items():
        save_fig(fig, str(plot_dir / f'pulse_lick_real_vs_rnn_{cond}'))
    save_fig(plot_kernel_real_vs_rnn(load_behavioural('elta'), sim['kernel']),
             str(plot_dir / 'kernel_real_vs_rnn'))
