from collections.abc import Iterable
from matplotlib import pyplot as plt
import numpy as np

from jaxdsp import training


def plot_filter(X, Y, Y_reference, Y_estimated, title):
    column_titles = (
        ["Target", "Reference Implementation", "Estimated"]
        if Y_reference is not None
        else ["Target", "Estimated"]
    )
    row_titles = ["Input", "Output"]
    Ys = [Y, Y_reference, Y_estimated] if Y_reference is not None else [Y, Y_estimated]
    _, axes = plt.subplots(2, len(column_titles), figsize=(14, 6))
    plt.suptitle(title, size=16)
    for ax, column_title in zip(axes[0], column_titles):
        ax.set_title(column_title)
    for ax, row_title in zip(axes[:, 0], row_titles):
        ax.set_ylabel(row_title, size="large")
    for i, ax_column in enumerate(axes.T):
        in_plot = ax_column[0]
        # in_plot.stem(X, basefmt=' ')
        in_plot.plot(X)
        in_plot.set_ylim([X.min() - 0.1, X.max() + 0.1])

        out_plot = ax_column[1]
        # out_plot.stem(Ys[i], basefmt=' ')
        out_plot.plot(Ys[i])
        out_plot.set_ylim([Ys[i].min() - 0.1, Ys[i].max() + 0.1])
    plt.tight_layout()


def plot_loss(loss_history):
    _, (plot) = plt.subplots(1, 1, figsize=(14, 3))
    plot.plot(loss_history)
    plot.set_xlabel("Batch")
    plot.set_ylabel("Loss")
    plot.set_title("Loss over time", size=16)
    plot.set_yscale("log", base=10)
    plot.autoscale(tight=True)


def plot_params_single(processor_name, params_target, params_history):
    param_groups = []  # list of lists of (key, label, scalar_param_value)
    # All non-list params go into the first (and potentially only) param group (column).
    single_params = [
        (key, key, param)
        for (key, param) in params_target.items()
        if not isinstance(param, Iterable)
    ]
    if len(single_params) > 0:
        param_groups.append(single_params)
    # Each list param gets its own param group (column).
    for (key, params) in [
        (key, params)
        for (key, params) in params_target.items()
        if isinstance(params, Iterable)
    ]:
        param_groups.append(
            [(key, "${}_{}$".format(key, i), param) for i, param in enumerate(params)]
        )
    num_rows, num_cols = max([len(param_group) for param_group in param_groups]), len(
        param_groups
    )
    _, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 2))
    if num_rows == 1 and num_cols == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, axis=1)
    plt.suptitle("Estimated parameters for {}".format(processor_name), size=16)
    for param_group_i, param_group in enumerate(param_groups):
        for param_i, (key, label, param) in enumerate(param_group):
            plot = axes[param_i][param_group_i]
            plot.set_title(label)
            plot.axhline(y=param, c="g", linestyle="--", label="Actual params")
            param_history = params_history[key]
            if isinstance(param_history[0], Iterable):
                param_history = np.asarray(param_history)[:, param_i]
            plot.plot(param_history, c="b", label="Estimated params")
            if param_group_i == 0:
                plot.set_ylabel("Value")
            if param_i == 0:
                plot.legend()
            if param_i == len(param_group) - 1:
                plot.set_xlabel("Batch")
            plot.autoscale()
    plt.tight_layout()


def plot_params(params_targets, params_histories, processor_names):
    for processor_name, params_target, params_history in zip(
        processor_names, params_targets, params_histories
    ):
        plot_params_single(
            processor_name,
            params_target,
            params_history,
        )


def plot_optimization(
    processor_config, Xs, params_inits, params_target, varying_param_name, steps=50
):
    """Used to investigate the shape of the gradient curve for a single target value across
    a range of different initial values.
    `params_inits` should be a list of dicts - one for each parameter tree to initialize with
    `params_target` should be a single target params dict
    """

    params_estimated = {key: [] for key in params_target.keys()}
    initial_losses = []
    losses = []
    x = [
        param_init[varying_param_name] for param_init in params_inits
    ]  # TODO only pass in one to vary
    for params_init in params_inits:
        trainer = training.IterativeTrainer(
            processor_config,
            training.Config(),
            params_init,
            track_history=True,
        )
        for i in range(steps):
            X = Xs[i % Xs.shape[0]]
            carry_target, Y_target = trainer.processor.tick_buffer(carry_target, X)
            trainer.step(X, Y_target)
        for key, param_estimated in params_estimated_i[processor.NAME].items():
            params_estimated[key].append(param_estimated)
        initial_losses.append(loss_history[0])
        losses.append(loss_history[-1])
    _, axes = plt.subplots(len(params_target) + 2, 1, figsize=(14, 12))
    for param_i, (label, param_target) in enumerate(params_target.items()):
        estimated_params_plot = axes[param_i]
        estimated_params_plot.plot(x, params_estimated[label], linewidth=3)
        estimated_params_plot.set_title("Estimated {}".format(label), size=18)
        estimated_params_plot.set_ylabel(
            "Estimated after {} steps".format(steps), size=11
        )
        estimated_params_plot.axhline(
            param_target, linestyle="--", c="r", label="Target index"
        )
        estimated_params_plot.legend()
    initial_loss_plot = axes[-2]
    initial_loss_plot.plot(x, initial_losses, linewidth=3)
    initial_loss_plot.set_title("Initial loss", size=18)
    initial_loss_plot.set_ylabel("Initial MSE loss", size=11)
    optimized_loss_plot = axes[-1]
    optimized_loss_plot.plot(x, losses, linewidth=3)
    optimized_loss_plot.set_title("Optimized loss", size=18)
    optimized_loss_plot.set_xlabel("Initial guess for {} value".format(label), size=16)
    optimized_loss_plot.set_ylabel("MSE loss after {} steps".format(steps), size=11)
    for axis in axes:
        axis.grid(True)
    plt.tight_layout()
