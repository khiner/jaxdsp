import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
from functools import partial

import sys
sys.path.append('./')
sys.path.append('./processors')
import serial_processors
from loss_fns import mse

def process(processor, params, X, *init_state_args):
    return processor.tick_buffer({'params': params, 'state': processor.init_state(*init_state_args)}, X)

def evaluate(params_estimated, params_target, processor, X, *init_state_args):
    Y_estimated = process(processor, params_estimated, X, *init_state_args)
    Y_target = process(processor, params_target, X, *init_state_args)
    return X, Y_estimated, Y_target

def train(processors, Xs, step_size=0.1, num_batches=100):
    processor = serial_processors
    params_target = processor.create_params_target(processors)
    params_history = {processor_key: {key: [param] for (key, param) in inner_init_params.items()} for (processor_key, inner_init_params) in processor.init_params(processors).items()}
    loss_history = np.zeros(num_batches)

    # TODO use [vmap](https://jax.readthedocs.io/en/latest/jax.html#jax.vmap)
    #   to efficiently map this loss function across batches
    def processor_loss(params, X):
        Y_target = process(processor, params_target, X, processors)
        Y = process(processor, params, X, processors)
        return mse(Y, Y_target)

    grad_fn = jit(value_and_grad(processor_loss))

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(processor.init_params(processors))
    for batch_i in range(num_batches):
        loss, gradient = grad_fn(get_params(opt_state), Xs[batch_i % Xs.shape[0]])
        opt_state = opt_update(batch_i, gradient, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        for (processor_key, processor_params) in new_params.items():
            for (param_key, param) in processor_params.items():
                params_history[processor_key][param_key].append(param)

    params_estimated = get_params(opt_state)
    return params_estimated, params_target, params_history, loss_history
