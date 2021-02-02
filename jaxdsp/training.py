import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
import collections.abc
from operator import itemgetter

from jaxdsp.processors import serial_processors
from jaxdsp.loss import mse, correlation
from jax.tree_util import tree_map, tree_multimap

### Non-batched ###


class IterativeTrainer():
    def __init__(self, processor, params_init=None, step_size=0.2):
        '''
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
        '''

        def loss(params, state, X, Y_target):
            carry, Y_estimated = processor.tick_buffer(
                {'params': params, 'state': state}, X)
            return mse(Y_estimated, Y_target), carry['state']

        self.step_num = 0
        self.loss = 0.0
        self.grad_fn = jit(value_and_grad(loss, has_aux=True))
        self.opt_init, self.opt_update, self.get_params = optimizers.sgd(
            step_size)
        self.opt_state = self.opt_init(params_init or processor.init_params())
        self.processor_state = processor.init_state()

    def step(self, X, Y_target):
        (self.loss, self.processor_state), self.grads = self.grad_fn(
            self.get_params(self.opt_state), self.processor_state, X, Y_target)
        self.opt_state = self.opt_update(
            self.step_num, self.grads, self.opt_state)
        self.step_num += 1

    def params(self):
        params = self.get_params(self.opt_state)
        return {key: float(value) for key, value in params.items()}

    def params_and_loss(self):
        return {
            'params': self.params(),
            'loss': float(self.loss),
        }

### Batched ###


@jit
def mean_loss_and_grads(loss, grads):
    return np.mean(loss), tree_map(lambda grad: np.mean(grad, axis=0), grads)


def evaluate(carry_estimated, carry_target, processor, X):
    carry_estimated, Y_estimated = processor.tick_buffer(carry_estimated, X)
    carry_target, Y_target = processor.tick_buffer(carry_target, X)
    return Y_estimated, Y_target

# TODO evaluation callback to build up loss/params history instead of baking it in here


def train(processors, Xs, step_size=0.2, num_batches=200, batch_size=32, params_init=None, params_target=None):
    processor = serial_processors
    params_target = params_target or processor.default_target_params(
        processors)

    def loss(params, X):
        carry_estimated = {'params': params,
                           'state': processor.init_state(processors)}
        carry_target = {'params': params_target,
                        'state': processor.init_state(processors)}
        carry_estimated, Y_estimated = processor.tick_buffer(
            carry_estimated, X)
        carry_target, Y_target = processor.tick_buffer(carry_target, X)
        return mse(Y_estimated, Y_target)

    params_init = params_init or processor.init_params(processors)
    params_history = tree_map(lambda param: [param], params_init)
    loss_history = np.zeros(num_batches)
    grad_fn = jit(vmap(value_and_grad(loss), in_axes=(None, 0), out_axes=0))
    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params_init)
    for batch_i in range(num_batches):
        Xs_batch = Xs[np.random.choice(Xs.shape[0], size=batch_size)]
        loss, grads = mean_loss_and_grads(
            *grad_fn(get_params(opt_state), Xs_batch))
        # TODO clip grads such that all params are in [min, max]?
        opt_state = opt_update(batch_i, grads, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        # params_history = tree_multimap(lambda h, *rest: h.append(rest[0]), params_history, new_params)
        for (processor_key, processor_params) in new_params.items():
            for (param_key, param) in processor_params.items():
                params_history[processor_key][param_key].append(param)

    params_estimated = get_params(opt_state)
    return params_estimated, params_target, params_history, loss_history
