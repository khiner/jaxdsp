import jax.numpy as jnp

import jaxdsp.ddsp.loss as ddsp_loss

spectral_loss_opts = ddsp_loss.MultiScaleSpectralOpts(
    loss_type='L2',
    mag_weight=0.0,
    delta_time_weight=0.0,
    delta_freq_weight=0.0,
    cumsum_freq_weight=1.0,
    logmag_weight=0.0,
)


def mse(X, Y):
    return ((Y - X) ** 2).mean()


def mae(X, Y):
    return jnp.abs(Y - X).mean()


def spectral(X, Y):
    if len(X.shape) == 1:
        X = jnp.expand_dims(X, 0)
    if len(Y.shape) == 1:
        Y = jnp.expand_dims(Y, 0)
    return ddsp_loss.multi_scale_spectral(X, Y, spectral_loss_opts)


def correlation(X, Y):
    """Based on https://github.com/numpy/numpy/issues/2310#issuecomment-314265666"""
    c = jnp.correlate(
        (X - jnp.mean(X)) / (jnp.std(X) * len(X)), (Y - jnp.mean(Y)) / (jnp.std(Y))
    )[0]
    return 1 - (1 + c) / 2


# TODO error-to-signal ratio (ESR)
#   MSE normalized by signal energy: "You need to be more accurate when your signal is very quiet"
