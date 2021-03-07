import functools

import jax.numpy as jnp
from jax import jit

from jaxdsp.ddsp import spectral_ops


def safe_log(X, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    return jnp.log(jnp.where(X <= eps, eps, X))


def norm(X):
    return jnp.sqrt((X ** 2).sum())


def distance(target, value, kind="L1", weights=1.0):
    difference = target - value
    kind = kind.upper()
    if kind == "L1":
        return jnp.mean(jnp.abs(difference * weights))
    elif kind == "L2":
        return jnp.mean(difference ** 2 * weights)
    elif kind == "COSINE":
        return ((target * weights) @ (value * weights).T) / (norm(target) * norm(value))
        # return tf.losses.cosine_distance(target, value, weights=weights, axis=-1)
    else:
        raise ValueError(f"Distance type ({kind}), must be 'L1, 'L2', or 'COSINE'")


# Doesn't support `loudness` from original ddsp
class MultiScaleSpectralOpts:
    def __init__(
        self,
        # Note: removing smaller fft sizes seems to get rid of some small non-convex "bumps"
        # in the loss curve for a sine wave with a target frequency param
        # fft_sizes=(2048, 1024, 512, 256, 128, 64),
        fft_sizes=(2048, 1024, 512, 256, 128),
        distance_type="L1",
        mag_weight=1.0,
        delta_time_weight=0.0,
        delta_freq_weight=0.0,
        cumsum_freq_weight=0.0,
        logmag_weight=0.0,
        name="spectral_loss",
    ):
        self.fft_sizes = fft_sizes
        self.distance_type = distance_type
        self.mag_weight = mag_weight
        self.delta_time_weight = delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.cumsum_freq_weight = cumsum_freq_weight
        self.logmag_weight = logmag_weight

        self.spectrogram_ops = []
        for size in self.fft_sizes:
            spectrogram_op = functools.partial(spectral_ops.compute_mag, size=size)
            self.spectrogram_ops.append(spectrogram_op)


def multi_scale_spectral(X, Y, opts, weights=None):
    """https://github.com/magenta/ddsp/blob/master/ddsp/losses.py#L132"""

    loss = 0.0
    diff = jnp.diff

    # Compute loss for each fft size.
    for loss_op in opts.spectrogram_ops:
        target_mag = loss_op(Y)
        value_mag = loss_op(X)

        # Add magnitude loss.
        if opts.mag_weight > 0:
            loss += opts.mag_weight * distance(
                target_mag, value_mag, opts.distance_type, weights=weights
            )

        if opts.delta_time_weight > 0:
            target = diff(target_mag, axis=1)
            value = diff(value_mag, axis=1)
            loss += opts.delta_time_weight * distance(
                target, value, opts.distance_type, weights=weights
            )

        if opts.delta_freq_weight > 0:
            target = diff(target_mag, axis=2)
            value = diff(value_mag, axis=2)
            loss += opts.delta_freq_weight * distance(
                target, value, opts.distance_type, weights=weights
            )

        # TODO(kyriacos) normalize cumulative spectrogram
        if opts.cumsum_freq_weight > 0:
            target = jnp.cumsum(target_mag, axis=2)
            value = jnp.cumsum(value_mag, axis=2)
            loss += opts.cumsum_freq_weight * distance(
                target, value, opts.distance_type, weights=weights
            )

        # Add logmagnitude loss, reusing spectrogram.
        if opts.logmag_weight > 0:
            target = safe_log(target_mag)
            value = safe_log(value_mag)
            loss += opts.logmag_weight * distance(
                target, value, opts.distance_type, weights=weights
            )

    return loss
