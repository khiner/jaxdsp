import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
import collections.abc

from jaxdsp.processors import serial_processors
from jaxdsp.loss import mse, correlation
from jax.tree_util import tree_map, tree_multimap

@jit
def mean_loss_and_grads(loss, grads):
    return np.mean(loss), tree_map(lambda grad: np.mean(grad, axis=0), grads)

def process(processor, params, X, *init_state_args):
    carry, Y = processor.tick_buffer({'params': params, 'state': processor.init_state(*init_state_args)}, X)
    return Y

def evaluate(params_estimated, params_target, processor, X, *init_state_args):
    Y_estimated = process(processor, params_estimated, X, *init_state_args)
    Y_target = process(processor, params_target, X, *init_state_args)
    return Y_estimated, Y_target

def train(processors, Xs, step_size=0.2, num_batches=200, batch_size=32,
          params_init=None, params_target=None):
    processor = serial_processors
    params_target = params_target or processor.default_target_params(processors)
    def loss(params, X):
        Y_estimated, Y_target = evaluate(params, params_target, processor, X, processors)
        return mse(Y_estimated, Y_target)

    params_init = params_init or processor.init_params(processors)
    params_history = tree_map(lambda param: [param], params_init)
    loss_history = np.zeros(num_batches)
    grad_fn = jit(vmap(value_and_grad(loss), in_axes=(None, 0), out_axes=0))
    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params_init)
    for batch_i in range(num_batches):
        Xs_batch = Xs[np.random.choice(Xs.shape[0], size=batch_size)]
        loss, grads = mean_loss_and_grads(*grad_fn(get_params(opt_state), Xs_batch))
        # TODO clip grads such that all params are in [0, 1]
        opt_state = opt_update(batch_i, grads, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        # params_history = tree_multimap(lambda h, *rest: h.append(rest[0]), params_history, new_params)
        for (processor_key, processor_params) in new_params.items():
            for (param_key, param) in processor_params.items():
                params_history[processor_key][param_key].append(param)

    params_estimated = get_params(opt_state)
    return params_estimated, params_target, params_history, loss_history
