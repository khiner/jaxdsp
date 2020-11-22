from collections.abc import Iterable
from matplotlib import pyplot as plt
import numpy as np

def plot_filter(X, Y, Y_reference, Y_estimated, title):
    column_titles = ['Target', 'Reference Implementation', 'Estimated'] if Y_reference is not None else ['Target', 'Estimated']
    row_titles = ['Input', 'Output']
    Ys = [Y, Y_reference, Y_estimated] if Y_reference is not None else [Y, Y_estimated]
    _, axes = plt.subplots(2, len(column_titles), figsize=(14, 6))
    plt.suptitle(title, size=16)
    for ax, column_title in zip(axes[0], column_titles):
        ax.set_title(column_title)
    for ax, row_title in zip(axes[:,0], row_titles):
        ax.set_ylabel(row_title, size='large')
    for i, ax_column in enumerate(axes.T):
        in_plot = ax_column[0]
        in_plot.stem(X, basefmt=' ')
        in_plot.set_ylim([X.min() - 0.1, X.max() + 0.1])

        out_plot = ax_column[1]
        out_plot.stem(Ys[i], basefmt=' ')
        out_plot.set_ylim([Ys[i].min() - 0.1, Ys[i].max() + 0.1])
    plt.tight_layout()

def plot_loss(loss_history):
    _, (plot) = plt.subplots(1, 1, figsize=(14, 3))
    plot.plot(loss_history)
    plot.set_xlabel('Batch')
    plot.set_ylabel('Loss')
    plot.set_title('Loss over time', size=16)
    plot.autoscale(tight=True)

def plot_params(params_target, params_history):
    param_groups = [] # list of lists of (key, label, scalar_param_value)
    # All non-list params go into the first (and potentially only) param group (column).
    single_params = [(key, key, param) for (key, param) in params_target.items() if not isinstance(param, Iterable)]
    if len(single_params) > 0: param_groups.append(single_params)
    # Each list param gets its own param group (column).
    for (key, params) in [(key, params) for (key, params) in params_target.items() if isinstance(params, Iterable)]:
        param_groups.append([(key, '${}_{}$'.format(key, i), param) for i, param in enumerate(params)])
    num_rows, num_cols = max([len(param_group) for param_group in param_groups]), len(param_groups)
    _, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 2))
    if len(axes.shape) == 1: axes = np.expand_dims(axes, axis=1)
    plt.suptitle('Estimated params over time', size=16)
    for param_group_i, param_group in enumerate(param_groups):
        for param_i, (key, label, param) in enumerate(param_group):
            plot = axes[param_i][param_group_i]
            plot.set_title(label)
            plot.set_xlabel('Batch')
            plot.axhline(y=param, c='g', linestyle='--', label='Actual params')
            param_history = params_history[key]
            if isinstance(param_history[0], Iterable): param_history = np.asarray(param_history)[:,param_i]
            plot.plot(param_history, c='b', label='Estimated params')
            if param_group_i == 0: plot.set_ylabel('Value')
            if param_i == 0: plot.legend()
            plot.autoscale(tight=True)
    plt.tight_layout()
