import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
from operator import itemgetter
from collections.abc import Iterable

from jaxdsp.processors import serial_processors
import jaxdsp.loss
from jax.tree_util import tree_map, tree_multimap


@jit
def mean_loss_and_grads(loss, grads):
    return np.mean(loss), tree_map(lambda grad: np.mean(grad, axis=0), grads)


class Config:
    def __init__(self, loss_fn=jaxdsp.loss.mse, step_size=0.2):
        self.step_size = step_size
        self.loss_fn = loss_fn


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
        training_config=Config(),
        processor_config=None,
        track_history=False,
    ):
        """
        from jaxdsp.training import IterativeTrainer

        processor = lbcf
        Xs = Xs_chirp
        carry_target = {'params': processor.default_target_params(), 'state': processor.init_state()}
        trainer = IterativeTrainer(processor, processor.init_params())
        for _ in range(10):
            X = Xs[np.random.randint(Xs.shape[0])]
            carry_target, Y_target = processor.tick_buffer(carry_target, X)
            trainer.step(X, Y_target)

        params_and_loss = trainer.params_and_loss()
        """

        def loss(params, state, X, Y_target):
            carry, Y_estimated = processor.tick_buffer(
                {"params": params, "state": state}, X
            )
            if Y_estimated.shape == Y_target.shape[::-1]:
                Y_estimated = Y_estimated.T  # TODO should eventually remove this check
            return training_config.loss_fn(Y_estimated, Y_target), carry["state"]

        processor_config = processor_config or processor.config()

        self.step_num = 0
        self.loss = 0.0
        # jit(vmap(value_and_grad(loss), in_axes=(None, 0), out_axes=0))
        self.grad_fn = jit(value_and_grad(loss, has_aux=True))
        self.opt_init, self.opt_update, self.get_params = optimizers.sgd(
            training_config.step_size
        )
        self.opt_state = self.opt_init(processor_config.params_init)
        self.processor_state = processor_config.state_init
        self.step_evaluator = (
            LossHistoryAccumulator(processor_config.params_init)
            if track_history
            else None
        )

    def step(self, X, Y_target):
        # loss, grads = mean_loss_and_grads(*grad_fn(get_params(opt_state), Xs_batch))
        (self.loss, self.processor_state), self.grads = self.grad_fn(
            self.get_params(self.opt_state), self.processor_state, X, Y_target
        )
        self.opt_state = self.opt_update(self.step_num, self.grads, self.opt_state)
        self.step_num += 1
        if self.step_evaluator:
            self.step_evaluator.after_step(self.loss, self.get_params(self.opt_state))

    def params(self):
        return float_params(self.get_params(self.opt_state))

    def params_and_loss(self):
        return {
            "params": self.params(),
            "loss": float(self.loss),
        }


def evaluate(carry_estimated, carry_target, processor, X):
    carry_estimated, Y_estimated = processor.tick_buffer(carry_estimated, X)
    carry_target, Y_target = processor.tick_buffer(carry_target, X)
    return Y_estimated, Y_target
