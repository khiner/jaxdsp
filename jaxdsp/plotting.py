from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt

from jaxdsp import training
from jaxdsp.params import params_to_float
from jaxdsp.training import TrainChartEventAccumulator


def plot_filter(fig, X, Y, Y_estimated, Y_reference, title):
    column_titles = (
        ["Target", "Reference Implementation", "Estimated"]
        if Y_reference is not None
        else ["Target", "Estimated"]
    )
    row_titles = ["Input", "Output"]
    Ys = [Y, Y_reference, Y_estimated] if Y_reference is not None else [Y, Y_estimated]
    axes = fig.subplots(2, len(column_titles), sharex=True)
    fig.suptitle(title, size=16)
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
        out_plot.plot(Ys[i].T)
        out_plot.set_ylim([Ys[i].min() - 0.1, Ys[i].max() + 0.1])


def plot_processor(fig, params_target, params_history):
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
    axes = fig.subplots(num_rows, num_cols, sharex=True)
    if num_rows == 1 and num_cols == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(axes.shape) == 1:
        axes = np.expand_dims(axes, axis=1)
    fig.suptitle(params_history["label"], fontsize="x-large")
    params_history_data = params_history["data"]
    for param_group_i, param_group in enumerate(param_groups):
        for param_i, (key, label, param) in enumerate(param_group):
            plot = axes[param_i][param_group_i]
            plot.set_title(label)
            plot.axhline(y=param, c="g", linestyle="--", label="Actual params")
            param_history_data = [
                params_history_i[key] for params_history_i in params_history_data
            ]
            if isinstance(param_history_data[0], Iterable):
                param_history_data = np.asarray(param_history_data)[:, param_i]
            plot.plot(param_history_data, c="b", label="Estimated params")
            if param_group_i == 0:
                plot.set_ylabel("Value")
            if param_i == 0:
                plot.legend()
            if param_i == len(param_group) - 1:
                plot.set_xlabel("Batch")
            plot.autoscale()


def plot_processors(
    fig, params_targets, params_histories, processor_names, is_row=True
):
    subfigs = fig.subfigures(
        1 if is_row else len(processor_names),
        len(processor_names) if is_row else 1,
        wspace=0.07,
    )

    if len(processor_names) == 1:
        subfigs = [subfigs]

    for subfig, processor_name, params_target, params_history in zip(
        subfigs, processor_names, params_targets, params_histories
    ):
        if isinstance(processor_name, list):
            # Nested processors are processed in parallel, so display in nested column
            plot_processors(
                subfig, params_target, params_history, processor_name, not is_row
            )
        else:
            plot_processor(subfig, params_target, params_history)


def plot_train(
    trainer,
    params,
    X,
    Y,
    Y_estimated,
    Y_reference,
    title=None,
    plot_loss_history=True,
    plot_params_history=True,
):
    num_rows = 1
    if plot_loss_history:
        num_rows += 1
    if plot_params_history:
        num_rows += 1

    fig = plt.figure(constrained_layout=True, figsize=(16, num_rows * 4))
    height_ratios = None
    if plot_loss_history:
        height_ratios = [0.75, 0.5, 1] if num_rows == 3 else [0.25, 1]
    subfigs = fig.subfigures(num_rows, 1, wspace=0.07, height_ratios=height_ratios)
    subfig_i = 0

    plot_filter(subfigs[subfig_i], X, Y, Y_estimated, Y_reference, title)
    subfig_i += 1

    accumulator = TrainChartEventAccumulator()
    for event in trainer.step_events:
        accumulator.accumulate(event)

    if plot_loss_history:
        plot = subfigs[subfig_i].subplots(1, 1)
        loss_series = accumulator.get_loss_series()
        plot.plot(loss_series["data"])
        plot.set_xlabel("Batch")
        plot.set_ylabel(loss_series["label"])
        plot.set_title("Loss over time", size=16)
        plot.set_yscale("log", base=10)
        plot.autoscale(tight=True)
        subfig_i += 1

    if plot_params_history:
        plot_processors(
            subfigs[subfig_i],
            params_to_float(params),
            accumulator.get_params_series()["data"],
            trainer.processor_names,
        )
        subfig_i += 1


# TODO fix and keep one example around, since this is neat
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
