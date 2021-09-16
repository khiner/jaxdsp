from jax import value_and_grad, jit

from jaxdsp import processor_graph
from jaxdsp.params import params_to_float
from jaxdsp.loss import LossOptions, loss_fn
from jaxdsp.optimizers import create_optimizer
from jaxdsp.processors import (
    processor_names_from_graph_config,
    graph_config_to_carry,
    params_to_unit_scale,
    params_from_unit_scale,
    get_graph_params,
    processor_names_to_graph_config,
    init_graph_state,
)
from jaxdsp.tracer import trace, time_ms


class TrainStepEvent:
    # `processor_names` and `params` must have the same shape (the shape of the processor graph).
    # They're only separate here since params are their own isolated thing from the perspective of the trainer.
    # Processor names are only used for series labels.
    def __init__(self, processor_names, params, loss):
        assert len(processor_names) == len(params)
        for inner_names, inner_params in zip(processor_names, params):
            assert len(inner_names) == len(inner_params)

        self.time_ms = time_ms()
        self.processor_names = processor_names
        self.params = params_to_float(params)
        self.loss = None if loss is None else float(loss)

    def serialize(self):
        return self.__dict__


class TrainChartEventAccumulator:
    def __init__(self):
        self.data = [
            {"id": "loss", "label": "Loss", "data": []},
            # "data" within the "params" series will itself be an array-of-arrays of series, with a series for each
            # processor (in the nested serial->parallel arrays format used for the processor graph throughout JAXdsp).
            # Note that the "params" series will be entirely cleared each time `accumulate` receives a graph with a
            # different topology.
            # The target use-case is tracking how a single graph optimizes over time. If you are interested in past
            # topologies for a single training run, just create an accumulator and `accumulate` up to the event
            # at the time of interest!
            # Also, there's nothing stopping a client from storing a time-series of full `data` instances for each
            # event, for e.g. scrobbling over full graph/parameter/loss states over time.
            {"id": "params", "label": "Params", "data": []},
        ]
        self.most_recent_processor_names = []

    def accumulate(self, event):
        self.get_loss_series()["data"].append(event.loss)
        # If the received event has a different graph topology, replace params series' with ones matching the new graph.
        if event.processor_names != self.most_recent_processor_names:
            self.get_params_series()["data"].clear()
            self.get_params_series()["data"] = [
                [
                    {"id": processor_name, "label": processor_name, "data": []}
                    for processor_name in processor_names_inner
                ]
                for processor_names_inner in event.processor_names
            ]

        for inner_params_data, inner_event_params in zip(
            self.get_params_series()["data"], event.params
        ):
            for processor_params_data, event_param_value in zip(
                inner_params_data, inner_event_params
            ):
                processor_params_data["data"].append(event_param_value)

        self.most_recent_processor_names = event.processor_names
        return self.data

    def get_loss_series(self):
        return self.data[0]

    def get_params_series(self):
        return self.data[1]


class IterativeTrainer:
    def __init__(
        self,
        graph_config=None,
        loss_options=None,
        optimizer_options=None,
        track_history=False,
    ):
        self.step_num = 0
        self.step_events = []
        self.loss = 0.0
        self.track_history = track_history
        self.optimizer = None
        self.grad_fn = None
        self.opt_state = None
        self.loss_options = None
        self.params, self.state = None, None
        self.processor_names = None
        self.set_optimizer_options(optimizer_options)
        self.set_graph_config(graph_config)
        self.set_loss_options(loss_options)

    def set_graph_config(self, graph_config):
        self.processor_names = processor_names_from_graph_config(graph_config)
        self.set_carry(graph_config_to_carry(graph_config))

    def set_carry(self, carry):
        self.step_events = []
        self.loss = None
        if carry:
            params, self.state = carry
            self.params = params or get_graph_params(
                processor_names_to_graph_config(self.processor_names)
            )
        else:
            self.params, self.state = None, None

        self.update_opt_state()

    def set_optimizer_options(self, optimizer_options):
        self.optimizer = (
            create_optimizer(
                optimizer_options.get("name"), optimizer_options.get("params")
            )
            if optimizer_options
            else create_optimizer()
        )

        if self.processor_names:
            self.state = init_graph_state(
                processor_names_to_graph_config(self.processor_names)
            )

        self.update_opt_state()

    def update_opt_state(self):
        self.opt_state = self.optimizer.init(
            params_to_unit_scale(self.params, self.processor_names)
        )

    def set_loss_options(self, loss_options):
        self.loss_options = loss_options or LossOptions()

        def processor_loss(unit_scale_params, state, X, Y_target):
            params = params_from_unit_scale(unit_scale_params, self.processor_names)
            carry, Y_estimated = processor_graph.tick_buffer(
                (params, state), X, self.processor_names
            )
            return (
                loss_fn(Y_estimated, Y_target, self.loss_options),
                carry[1],  # return state as aux
            )

        self.grad_fn = jit(value_and_grad(processor_loss, has_aux=True))

    @trace
    def step(self, X, Y_target):
        if not self.processor_names:
            return

        params_unit = params_to_unit_scale(self.params, self.processor_names)
        (self.loss, self.state), self.grads = self.grad_fn(
            params_unit,
            self.state,
            X,
            Y_target,
        )
        self.opt_state = self.optimizer.update(
            self.step_num, self.grads, self.opt_state
        )
        self.step_num += 1
        self.params = params_from_unit_scale(
            self.optimizer.get_params(self.opt_state), self.processor_names
        )
        self.append_step_event()

    def append_step_event(self):
        self.step_events.append(
            TrainStepEvent(self.processor_names, self.params, self.loss)
        )

    def get_events(self):
        return self.step_events

    def get_events_serialized(self):
        return [event.serialize() for event in self.step_events]

    def clear_events(self):
        self.step_events.clear()


def evaluate(carry_estimated, carry_target, processor, X):
    carry_estimated, Y_estimated = processor.tick_buffer(carry_estimated, X)
    carry_target, Y_target = processor.tick_buffer(carry_target, X)
    return Y_estimated, Y_target
