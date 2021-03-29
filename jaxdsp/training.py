import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
from operator import itemgetter
from collections.abc import Iterable

from jaxdsp.processors import serial_processors
from jaxdsp.loss import LossOptions, loss_fn
from jaxdsp.optimizers import OptimizationOptions, default_optimization_options
from jax.tree_util import tree_map, tree_multimap


default_loss_options = LossOptions(
    weights={
        "sample": 1.0,
    },
    distance_types={
        "sample": "L2",
    },
)

@jit
def mean_loss_and_grads(loss, grads):
    return np.mean(loss), tree_map(lambda grad: np.mean(grad, axis=0), grads)


class LossHistoryAccumulator:
    def __init__(self, params_init):
        self.params_history = tree_map(lambda param: [param], params_init)
        self.loss_history = []

    def after_step(self, loss, new_params):
        self.loss_history.append(loss)
        self.params_history = tree_multimap(
            lambda new_param, params: params + [new_param],
            new_params,
            self.params_history,
        )


def float_params(params):
    return tree_map(
        lambda param: param
        if hasattr(param, "ndim") and param.ndim > 0
        else float(param),
        params,
    )


class IterativeTrainer:
    def __init__(
        self,
        processor,
        loss_options=None,
        optimization_options=None,
        processor_config=None,
        track_history=False,
    ):
        self.processor = processor
        self.step_num = 0
        self.loss = 0.0
        self.processor_config = processor_config or processor.config()
        self.current_params = self.processor_config.params_init
        self.step_evaluator = (
            LossHistoryAccumulator(self.processor_config.params_init)
            if track_history
            else None
        )
        self.set_optimization_options(
            optimization_options or default_optimization_options
        )
        self.set_loss_options(loss_options or default_loss_options)

    def set_optimization_options(self, optimization_options):
        (
            self.opt_init,
            self.opt_update,
            self.get_params,
        ) = optimization_options.create()
        self.opt_state = self.opt_init(self.current_params)
        self.processor_state = self.processor_config.state_init

    def set_loss_options(self, loss_options):
        def processor_loss(params, state, X, Y_target):
            carry, Y_estimated = self.processor.tick_buffer(
                {"params": params, "state": state}, X
            )
            if Y_estimated.shape == Y_target.shape[::-1]:
                Y_estimated = Y_estimated.T  # TODO should eventually remove this check
            return (
                loss_fn(Y_estimated, Y_target, loss_options),
                carry["state"],
            )

        # jit(vmap(value_and_grad(processor_loss), in_axes=(None, 0), out_axes=0))
        self.grad_fn = jit(value_and_grad(processor_loss, has_aux=True))

    def step(self, X, Y_target):
        # loss, grads = mean_loss_and_grads(*grad_fn(get_params(opt_state), Xs_batch))
        (self.loss, self.processor_state), self.grads = self.grad_fn(
            self.current_params,
            self.processor_state,
            X,
            Y_target,
        )
        self.opt_state = self.opt_update(self.step_num, self.grads, self.opt_state)
        self.step_num += 1
        self.current_params = self.get_params(self.opt_state)
        if self.step_evaluator:
            self.step_evaluator.after_step(self.loss, self.current_params)

    def params(self):
        return float_params(self.current_params)

    def params_and_loss(self):
        return {
            "params": self.params(),
            "loss": float(self.loss),
        }


def evaluate(carry_estimated, carry_target, processor, X):
    carry_estimated, Y_estimated = processor.tick_buffer(carry_estimated, X)
    carry_target, Y_target = processor.tick_buffer(carry_target, X)
    return Y_estimated, Y_target
