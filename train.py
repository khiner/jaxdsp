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

def train(processor, X, step_size=0.1, num_batches=100):
    params_target = processor.create_params_target()
    params_history = {key: [param] for (key, param) in processor.init_params().items()}
    loss_history = np.zeros(num_batches)
    Y_target = process(processor, params_target, X)

    def processor_loss(params):
        Y = process(processor, params, X)
        return mse(Y, Y_target)

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(processor.init_params())
    grad_fn = jit(value_and_grad(processor_loss))
    for batch_i in range(num_batches):
        loss, gradient = grad_fn(get_params(opt_state))
        opt_state = opt_update(batch_i, gradient, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        for param_key in new_params.keys():
            params_history[param_key].append(new_params[param_key])

    params_estimated = get_params(opt_state)
    Y_estimated = process(processor, params_estimated, X)
    return params_estimated, params_target, Y_estimated, Y_target, params_history, loss_history


def train_serial(processors, X, step_size=0.1, num_batches=100):
    processor = serial_processors
    params_target = processor.create_params_target(processors)
    params_history = {processor_key: {key: [param] for (key, param) in inner_init_params.items()} for (processor_key, inner_init_params) in processor.init_params(processors).items()}
    loss_history = np.zeros(num_batches)
    Y_target = process(processor, params_target, X, processors)

    def processor_loss(params):
        Y = process(processor, params, X, processors)
        return mse(Y, Y_target)

    grad_fn = jit(value_and_grad(processor_loss))

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(processor.init_params(processors))
    for batch_i in range(num_batches):
        loss, gradient = grad_fn(get_params(opt_state))
        opt_state = opt_update(batch_i, gradient, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        for (processor_key, processor_params) in new_params.items():
            for (param_key, param) in processor_params.items():
                params_history[processor_key][param_key].append(param)

    params_estimated = get_params(opt_state)
    Y_estimated = process(processor, params_estimated, X, processors)
    return params_estimated, params_target, Y_estimated, Y_target, params_history, loss_history
