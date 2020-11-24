import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
from functools import partial

import sys
sys.path.append('./')
sys.path.append('./processors')
from serial_processors import SerialProcessors
from process import process, process_serial
from loss_fns import processor_loss, processor_loss_serial

def train(processor_class, X, step_size=0.1, num_batches=100):
    params_target = processor_class.create_params_target()
    params_history = {key: [param] for (key, param) in processor_class.init_params().items()}
    loss_history = np.zeros(num_batches)
    Y_target = process(params_target, processor_class, X)
    processor_class_loss = partial(processor_loss, processor_class=processor_class, X=X, Y_target=Y_target)

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(processor_class.init_params())
    grad_fn = jit(value_and_grad(processor_class_loss))
    for batch_i in range(num_batches):
        loss, gradient = grad_fn(get_params(opt_state))
        opt_state = opt_update(batch_i, gradient, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        for param_key in new_params.keys():
            params_history[param_key].append(new_params[param_key])

    params_estimated = get_params(opt_state)
    Y_estimated = process(params_estimated, processor_class, X)
    return params_estimated, params_target, Y_estimated, Y_target, params_history, loss_history


def train_serial(processor_classes, X, step_size=0.1, num_batches=100):
    processor_class = SerialProcessors
    params_target = processor_class.create_params_target(processor_classes)
    params_history = {processor_key: {key: [param] for (key, param) in inner_init_params.items()} for (processor_key, inner_init_params) in processor_class.init_params(processor_classes).items()}
    loss_history = np.zeros(num_batches)
    Y_target = process_serial(params_target, processor_class, processor_classes, X)
    processor_class_loss_serial = partial(processor_loss_serial, processor_classes=processor_classes, processor_class=processor_class, X=X, Y_target=Y_target)
    grad_fn = jit(value_and_grad(processor_class_loss_serial))

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(processor_class.init_params(processor_classes))
    for batch_i in range(num_batches):
        loss, gradient = grad_fn(get_params(opt_state))
        opt_state = opt_update(batch_i, gradient, opt_state)
        loss_history[batch_i] = loss
        new_params = get_params(opt_state)
        for (processor_key, processor_params) in new_params.items():
            for (param_key, param) in processor_params.items():
                params_history[processor_key][param_key].append(param)

    params_estimated = get_params(opt_state)
    Y_estimated = process_serial(params_estimated, processor_class, processor_classes, X)
    return params_estimated, params_target, Y_estimated, Y_target, params_history, loss_history
