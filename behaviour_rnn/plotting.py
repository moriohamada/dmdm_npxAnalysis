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

def plot_example_trials(model, df_subj, pos_weight, n_examples=3, seed=0,
                        config=ANALYSIS_OPTIONS):
    """grid of example trials per (block, outcome): TF + RNN p_lick + lick/change markers.
    rows = early hit / early FA / early miss / late hit / late FA / late miss.
    cols = n_examples random trials per cell."""
    from behaviour_rnn.simulate_behaviour import predict_for_df
    p_lick, tf_in, meta = predict_for_df(model, df_subj, pos_weight)
    df = meta['df']
    dt = meta['dt']
    max_t = p_lick.shape[1]
    t_grid = np.arange(max_t) * dt

    rng = np.random.default_rng(seed)
    outcomes = [('hit',  meta['is_hit']),
                ('fa',   meta['is_fa']),
                ('miss', meta['is_miss'])]
    blocks_def = [('early', 1.0), ('late', -1.0)]

    rows, row_labels = [], []
    for block_name, block_val in blocks_def:
        for outcome_name, outcome_mask in outcomes:
            sel = np.where((meta['blocks'] == block_val) & outcome_mask)[0]
            picks = (rng.choice(sel, size=min(n_examples, len(sel)), replace=False)
                     if len(sel) else np.array([], dtype=int))
            rows.append(picks)
            row_labels.append(f'{block_name} {outcome_name}')

    n_rows = len(rows)
    fig = make_subplots(rows=n_rows, cols=n_examples,
                        specs=[[{'secondary_y': True}] * n_examples] * n_rows,
                        row_titles=row_labels,
                        shared_xaxes=False, shared_yaxes=False,
                        horizontal_spacing=0.06, vertical_spacing=0.025)

    for r, trial_ids in enumerate(rows, start=1):
        block_name = 'early' if r <= 3 else 'late'
        col = _block_colour(block_name)
        for c, trial_i in enumerate(trial_ids, start=1):
            tf_tr = tf_in[trial_i]
            pl_tr = p_lick[trial_i]
            valid = ~np.isnan(pl_tr)
            if not valid.any():
                continue
            t = t_grid[valid]

            fig.add_trace(go.Scatter(x=t, y=tf_tr[valid], mode='lines',
                line=dict(color='dimgrey', width=1.2),
                showlegend=(r == 1 and c == 1), name='TF (oct)'),
                row=r, col=c, secondary_y=False)
            fig.add_trace(go.Scatter(x=t, y=pl_tr[valid], mode='lines',
                line=dict(color=col, width=1.6),
                showlegend=(r == 1 and c == 1), name='RNN P(lick)'),
                row=r, col=c, secondary_y=True)

            tr = df.iloc[int(trial_i)]
            stim_t = float(tr['stimT']) if np.isfinite(tr.get('stimT', np.nan)) else None
            if meta['is_hit'][trial_i] or meta['is_miss'][trial_i]:
                if stim_t is not None:
                    fig.add_vline(x=stim_t, line_dash='dash', line_color='black',
                                  row=r, col=c)
            if meta['is_hit'][trial_i] and np.isfinite(tr.get('rt_RT', np.nan)):
                fig.add_vline(x=stim_t + float(tr['rt_RT']),
                              line_dash='solid', line_color='seagreen',
                              row=r, col=c)
            if meta['is_fa'][trial_i] and np.isfinite(tr.get('rt_FA', np.nan)):
                fig.add_vline(x=float(tr['rt_FA']),
                              line_dash='solid', line_color='crimson',
                              row=r, col=c)

    fig.update_xaxes(title_text='time in trial (s)', row=n_rows)
    fig.update_yaxes(title_text='TF (octaves)', secondary_y=False, col=1)
    fig.update_yaxes(title_text='P(lick)', secondary_y=True, col=n_examples)
    fig.update_yaxes(range=[0, 1], secondary_y=True)
    fig.update_layout(template='plotly_white',
                      height=180 * n_rows, width=320 * n_examples,
                      showlegend=True,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='right', x=1))
    return fig


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
                 str(plot_dir / 'hazard'))

    real_p = {subj: real_cached['pulse'][subj]} if subj in real_cached['pulse'] else {}
    sim_p  = {subj: sim['pulse']}
    if real_p:
        for cond, fig in plot_pulse_lick_real_vs_rnn(real_p, sim_p, config).items():
            save_fig(fig, str(plot_dir / f'pulse_{cond}'))

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
                 str(plot_dir / 'kernel'))


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
             str(plot_dir / 'hazard_real_vs_rnn'))
    for cond, fig in plot_pulse_lick_real_vs_rnn(
            load_behavioural('pulse_lick_prob'), sim['pulse']).items():
        save_fig(fig, str(plot_dir / f'pulse_lick_real_vs_rnn_{cond}'))
    save_fig(plot_kernel_real_vs_rnn(load_behavioural('elta'), sim['kernel']),
             str(plot_dir / 'kernel_real_vs_rnn'))
    save_fig(plot_outcome_dist_cohort(sim['outcome']),
             str(plot_dir / 'outcome_real_vs_rnn'))
