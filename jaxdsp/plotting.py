from collections.abc import Iterable

import numpy as np
from matplotlib import pyplot as plt

import jax.numpy as jnp
from jaxdsp import training
from jaxdsp.params import params_to_float
from jaxdsp.training import TrainChartEventAccumulator


def plot_signal_transform(fig, X, Y, Y_estimated, Y_reference):
    column_titles = (
        ["Input", "Target"]
        + (["Reference Implementation"] if Y_reference is not None else [])
        + ["Estimated"]
    )
    axes = fig.subplots(len(column_titles), 1, sharex=True)
    fig.suptitle("Signal Transform", size=16)
    for ax, row_title in zip(axes, column_titles):
        ax.set_ylabel(row_title, size="large")

    Ys = [X, Y] + ([Y_reference] if Y_reference is not None else []) + [Y_estimated]
    for i, plot in enumerate(axes):
        # plot.stem(Ys[i], basefmt=' ')
        plot.plot(Ys[i].T)
        plot.set_ylim([Ys[i].min() - 0.1, Ys[i].max() + 0.1])
        plot.autoscale(tight=True)


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
    for key, params in [
        (key, params)
        for (key, params) in params_target.items()
        if isinstance(params, Iterable)
    ]:
        param_groups.append([(key, f"${key}_{i}$", p) for i, p in enumerate(params)])
    num_rows, num_cols = max([len(pg) for pg in param_groups]), len(param_groups)
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
            plot.autoscale(tight=True)


def plot_processor_params(
    fig, params_targets, params_histories, processor_names, is_row=True, is_main=True
):
    subfigs = fig.subfigures(
        1 if is_row else len(processor_names),
        len(processor_names) if is_row else 1,
        wspace=0.07,
    )

    if len(processor_names) == 1:
        subfigs = [subfigs]

    if is_main:
        fig.suptitle("Processor Graph Params", fontsize=16)

    for subfig, processor_name, params_target, params_history in zip(
        subfigs, processor_names, params_targets, params_histories
    ):
        if isinstance(processor_name, list):
            # Nested processors are processed in parallel, so display in nested column
            plot_processor_params(
                subfig, params_target, params_history, processor_name, not is_row, False
            )
        else:
            plot_processor(subfig, params_target, params_history)


def get_num_rows(params):
    num_rows = 0
    if isinstance(params, dict):
        if len(params) > 0 and all(isinstance(v, Iterable) for v in params.values()):
            num_rows += max(len(v) for v in params.values())
        else:
            num_rows += sum(get_num_rows(v) for v in params.values())
    elif isinstance(params, Iterable):
        num_rows += max(get_num_rows(v) for v in params)
    else:
        num_rows += 1
    return num_rows


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
    num_param_rows = 0 if not plot_params_history else get_num_rows(params)
    num_rows = int(plot_params_history) + int(plot_loss_history) + 1
    height_ratios = (
        [num_param_rows]
        + [1] * int(plot_loss_history)
        + [2 if Y_reference is None else 3]
    )

    fig = plt.figure(constrained_layout=True, figsize=(16, 2 * sum(height_ratios)))
    subfigs = fig.subfigures(num_rows, 1, wspace=0.07, height_ratios=height_ratios)

    fig.suptitle(title, size=16)

    accumulator = TrainChartEventAccumulator()
    for event in trainer.step_events:
        accumulator.accumulate(event)

    subfig_i = 0

    if plot_params_history:
        plot_processor_params(
            subfigs[subfig_i],
            params_to_float(params),
            accumulator.get_params_series()["data"],
            trainer.processor_names,
        )
        subfig_i += 1

    if plot_loss_history:
        plot = subfigs[subfig_i].subplots(1, 1)
        loss_series = accumulator.get_loss_series()
        plot.plot(loss_series["data"], linewidth=2)
        plot.set_xlabel("Batch")
        plot.set_ylabel(loss_series["label"])
        plot.set_title("Loss over time", size=16)
        plot.set_yscale("log", base=10)
        plot.autoscale(tight=True)
        subfig_i += 1

    plot_signal_transform(subfigs[subfig_i], X, Y, Y_estimated, Y_reference)
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
        estimated_params_plot.set_title(f"Estimated {label}", size=18)
        estimated_params_plot.set_ylabel(f"Estimated after {steps} steps", size=11)
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
    optimized_loss_plot.set_xlabel(f"Initial guess for {label} value", size=16)
    optimized_loss_plot.set_ylabel(f"MSE loss after {steps} steps", size=11)
    for axis in axes:
        axis.grid(True)
    plt.tight_layout()
