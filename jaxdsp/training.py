import numpy as np
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, vmap
from jax.experimental import optimizers
import collections.abc

from jaxdsp.processors import serial_processors
from jaxdsp.loss import mse, correlation
from jax.tree_util import tree_map, tree_multimap

### Non-batched ###

# TODO refactor non-batched helper methods to support chunked continuous input,
# carrying state forward for both target and estimated processor state
def train_init(processor, params_init, step_size=0.2):
    '''
    from jaxdsp.training import train_init, train_step, params_from_train_state

    carry_target = {'params': processor.default_target_params(), 'state': processor.init_state()}
    train_state = train_init(processor, processor.default_target_params())
    for _ in range(100):
        X = Xs[np.random.randint(Xs.shape[0])]
        carry_target, Y_target = processor.tick_buffer(carry_target, X)
        train_state = train_step(X, Y_target, *train_state)

    params = params_from_train_state(*train_state)
    '''
    def loss(params, state, X, Y_target):
        carry, Y_estimated = processor.tick_buffer({'params': params, 'state': state}, X)
        return mse(Y_estimated, Y_target), carry['state']

    grad_fn = jit(value_and_grad(loss, has_aux=True))
    opt_init, opt_update, get_params = optimizers.sgd(step_size)
    opt_state = opt_init(params_init)
    step = 0
    return step, processor.init_state(), grad_fn, get_params, opt_update, opt_state

def train_step(X, Y_target, step, processor_state, grad_fn, get_params, opt_update, opt_state):
    (loss, processor_state), grads = grad_fn(get_params(opt_state), processor_state, X, Y_target)
    opt_state = opt_update(step, grads, opt_state)
    return step + 1, processor_state, grad_fn, get_params, opt_update, opt_state

def params_from_train_state(step, processor_state, grad_fn, get_params, opt_update, opt_state):
    params = get_params(opt_state)
    return {key: float(value) for key, value in params.items()}


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
    params_target = params_target or processor.default_target_params(processors)
    def loss(params, X):
        carry_estimated = {'params': params, 'state': processor.init_state(processors)}
        carry_target = {'params': params_target, 'state': processor.init_state(processors)}
        carry_estimated, Y_estimated = processor.tick_buffer(carry_estimated, X)
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
        loss, grads = mean_loss_and_grads(*grad_fn(get_params(opt_state), Xs_batch))
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
