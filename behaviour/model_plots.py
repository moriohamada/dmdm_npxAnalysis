"""
visualisation for leaky integrator model fits
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_integrator_fits(params, likelihoods, best_params,
                              params_to_explore=('threshold', 'tau')):
    """heatmaps of likelihoods across parameter space"""
    subjs = params['early'].keys()
    n_subj = len(subjs)
    fig = make_subplots(rows=n_subj, cols=2, shared_xaxes=True, shared_yaxes=True)

    for s, subj in enumerate(subjs):
        for col, block in enumerate(['early', 'late']):
            param_list = params[block][subj]
            ll = likelihoods[block][subj]

            df_grid = pd.DataFrame(param_list)
            df_grid['ll'] = ll
            profiled = df_grid.groupby(list(params_to_explore))['ll'].mean().reset_index()

            pivot = profiled.pivot(index=params_to_explore[0],
                                   columns=params_to_explore[1],
                                   values='ll')

            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=np.log10(pivot.columns.tolist()),
                y=pivot.index.tolist(),
                showscale=True, colorscale='Viridis',
            ), row=s + 1, col=col + 1)

            best = best_params[block][subj]
            fig.add_trace(go.Scatter(
                x=np.log10([best[params_to_explore[1]]]),
                y=[best[params_to_explore[0]]],
                mode='markers',
                marker={'symbol': 'cross', 'size': 10, 'color': 'white',
                        'line': dict(color='white', width=1)},
                showlegend=False,
            ), row=s + 1, col=col + 1)

    fig.update_layout(height=100 * n_subj, width=300)
    return fig


def visualize_best_params(best_params):
    param_keys = ['threshold', 'gain', 'tau', 'sigma']
    subjs = list(best_params['early'].keys())

    fig = make_subplots(rows=1, cols=4, subplot_titles=param_keys)

    for p, param in enumerate(param_keys):
        for subj in subjs:
            y = [best_params['early'][subj][param],
                 best_params['late'][subj][param]]
            fig.add_trace(go.Scatter(
                x=['early', 'late'], y=y,
                mode='lines+markers', name=subj,
                showlegend=(p == 0),
            ), row=1, col=p + 1)

    fig.update_layout(height=300, width=200 * len(param_keys))
    return fig
