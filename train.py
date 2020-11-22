import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad
from jax.experimental import optimizers

import sys
sys.path.append('./')
sys.path.append('./processors')
from serial_processors import SerialProcessors
from process import process, process_serial
from loss_fns import processor_loss, processor_loss_serial

def train(processor_class, X, step_size=0.1, num_batches=10):
    processor = processor_class()
    params_target = processor.create_params_target()
    params_history = {key: [param] for (key, param) in processor.init_params.items()}
    loss_history = np.zeros(num_batches)
    Y = jnp.zeros(X.size)
    Y_target = process(params_target, processor_class, X, Y)

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(processor.init_params)
    for batch_i in range(num_batches):
        loss, gradient = value_and_grad(processor_loss)(get_params(opt_state), process, processor_class, X, Y, Y_target)
        opt_state = opt_update(batch_i, gradient, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        for param_key in new_params.keys():
            params_history[param_key].append(new_params[param_key])

    params_estimated = get_params(opt_state)
    Y_estimated = process(params_estimated, processor_class, X, Y)
    return params_estimated, params_target, Y_estimated, Y_target, params_history, loss_history


def train_serial(processors, X, step_size=0.1, num_batches=10):
    processor_class = SerialProcessors
    processor = processor_class(processors)
    params_target = processor.create_params_target()
    params_history = {processor_key: {key: [param] for (key, param) in inner_init_params.items()} for (processor_key, inner_init_params) in processor.init_params.items()}
    loss_history = np.zeros(num_batches)
    Y = jnp.zeros(X.size)
    Y_target = process_serial(params_target, processor_class, processors, X, Y)

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(processor.init_params)
    for batch_i in range(num_batches):
        loss, gradient = value_and_grad(processor_loss_serial)(get_params(opt_state), process_serial, processor_class, processors, X, Y, Y_target)
        opt_state = opt_update(batch_i, gradient, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        for (processor_key, processor_params) in new_params.items():
            for (param_key, param) in processor_params.items():
                params_history[processor_key][param_key].append(param)

    params_estimated = get_params(opt_state)
    Y_estimated = process_serial(params_estimated, processor_class, processors, X, Y)
    return params_estimated, params_target, Y_estimated, Y_target, params_history, loss_history
