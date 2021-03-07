import functools

import jax.numpy as jnp
from jax import jit
import jax_spectral

# Port of https://github.com/magenta/ddsp/blob/master/ddsp/spectral_ops.py#L33,
# but using https://github.com/cifkao/jax-spectral instead of tf.signal.stft
# Note: keep an eye out for jax to implement this in
# [jax.scipy.signal](https://jax.readthedocs.io/en/latest/jax.scipy.html#module-jax.scipy.signal)
def stft(audio, sample_rate=16000, frame_size=2048, overlap=0.75, pad_end=True):
    assert frame_size * overlap % 2.0 == 0.0

    return jax_spectral.spectral.stft(
        audio,
        # sample_rate, # TODO need this
        nperseg=int(frame_size),
        noverlap=int(overlap),
        # nfft=int(frame_size),
        padded=pad_end,
    )


# Port of https://github.com/magenta/ddsp/blob/master/ddsp/spectral_ops.py#L76
def magnitute_spectrogram(audio, size=2048, overlap=0.75, pad_end=True):
    return jnp.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end)[2])


def safe_log(X, eps=1e-5):
    """Avoid taking the log of a non-positive number."""
    return jnp.log(jnp.where(X <= eps, eps, X))


def norm(X):
    return jnp.sqrt((X ** 2).sum())


def distance(target, value, kind="L1", weights=1.0):
    difference = target - value
    kind = kind.upper()
    weights = weights or 1.0
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
            self.spectrogram_ops.append(
                functools.partial(magnitute_spectrogram, size=size)
            )


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


spectral_loss_opts = MultiScaleSpectralOpts(
    distance_type="L2",
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
    return multi_scale_spectral(X, Y, spectral_loss_opts)


def correlation(X, Y):
    """Based on https://github.com/numpy/numpy/issues/2310#issuecomment-314265666"""
    c = jnp.correlate(
        (X - jnp.mean(X)) / (jnp.std(X) * len(X)), (Y - jnp.mean(Y)) / (jnp.std(Y))
    )[0]
    return 1 - (1 + c) / 2


# TODO error-to-signal ratio (ESR)
#   MSE normalized by signal energy: "You need to be more accurate when your signal is very quiet"
