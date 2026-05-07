"""
figure saving utilities
"""


def save_fig(fig, path_stem, formats=('png', 'pdf')):
    """save a plotly figure in multiple formats"""
    for ext in formats:
        fig.write_image(f'{path_stem}.{ext}')
