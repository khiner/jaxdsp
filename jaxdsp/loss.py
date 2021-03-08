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


def distance(X, Y, kind="L1"):
    kind = kind.upper()
    if kind == "L1":
        return jnp.mean(jnp.abs(Y - X))
    elif kind == "L2":
        return jnp.mean((Y - X) ** 2)
    elif kind == "COSINE":
        return (Y @ X.T) / (norm(Y) * norm(X))
    else:
        raise ValueError(f"Distance type ({kind}), must be 'L1, 'L2', or 'COSINE'")


loss_fn_for_label = {
    # Weight to compare linear magnitudes of spectrograms.
    # Core audio similarity loss. More sensitive to peak magnitudes than log magnitudes.
    "magnitude": lambda X_mag, Y_mag, distance_type: distance(
        X_mag, Y_mag, distance_type
    ),
    # Weight to compare log magnitudes of spectrograms.
    # Core audio similarity loss. More sensitive to quiet magnitudes than linear magnitudes.
    "log_magnitude": lambda X_mag, Y_mag, distance_type: distance(
        safe_log(X_mag), safe_log(Y_mag), distance_type
    ),
    # Weight to compare the first finite difference of spectrograms in time.
    # Emphasizes changes of magnitude in time, such as at transients.
    "delta_time": lambda X_mag, Y_mag, distance_type: distance(
        jnp.diff(X_mag, axis=1), jnp.diff(Y_mag, axis=1), distance_type
    ),
    # Weight to compare the first finite difference of spectrograms in frequency.
    # Emphasizes changes of magnitude in frequency, such as at the boundaries of a stack of harmonics.
    "delta_freq": lambda X_mag, Y_mag, distance_type: distance(
        jnp.diff(X_mag, axis=2), jnp.diff(Y_mag, axis=2), distance_type
    ),
    # Weight to compare the cumulative sum of spectrograms across frequency for each slice in time.
    # Similar to a 1-D Wasserstein loss.
    # This hopefully provides a non-vanishing gradient to push two non-overlapping sinusoids towards each other.
    "cumsum_freq": lambda X_mag, Y_mag, distance_type: distance(
        jnp.cumsum(X_mag, axis=2), jnp.cumsum(Y_mag, axis=2), distance_type
    ),
}

# Spectral losses based on https://github.com/magenta/ddsp/blob/master/ddsp/losses.py#L132
# Doesn't support `loudness` from original ddsp.
# ddsp also has lots more experimental losses (that are worth investigating at some point!)
class LossOpts:
    def __init__(
        self,
        weights={
            "magnitude": 1.0,
            "delta_time": 0.0,
            "delta_freq": 0.0,
            "cumsum_freq": 0.0,
            "log_magnitude": 0.0,
        },
        # Note: removing smaller fft sizes seems to get rid of some small non-convex "bumps"
        # in the loss curve for a sine wave with a target frequency param
        # fft_sizes=(2048, 1024, 512, 256, 128, 64),
        fft_sizes=(2048, 1024, 512, 256, 128),
        spectral_distance_type="L1",
    ):
        """Constructor, set loss weights of various components.
        Args:
        weights: Dict of loss labels to relative weighting of that loss.
            (See `loss_fn_for_label` above for details on loss types.)
        fft_sizes: Compare spectrograms at each of this list of fft sizes.
            Each spectrogram has a time-frequency resolution trade-off based on fft size,
            so comparing multiple scales allows multiple resolutions.
        spectral_distance_type: One of 'L1', 'L2', or 'COSINE'.
        """
        self.spectral_losses = [
            (
                weight,
                functools.partial(
                    loss_fn_for_label[label], distance_type=spectral_distance_type
                ),
            )
            for label, weight in weights.items()
            if weight and weight > 0
        ]
        self.spectrogram_fns = [
            functools.partial(magnitute_spectrogram, size=size) for size in fft_sizes
        ]
        self.spectral_distance_type = spectral_distance_type


# Based on https://github.com/magenta/ddsp/blob/master/ddsp/losses.py#L132
def multi_scale_spectral(X, Y, opts):
    loss = 0.0
    for spectrogram_fn in opts.spectrogram_fns:
        X_mag = spectrogram_fn(X)
        Y_mag = spectrogram_fn(Y)
        loss += sum(
            weight * loss_fn(X_mag, Y_mag) for (weight, loss_fn) in opts.spectral_losses
        )
    return loss


spectral_loss_opts = LossOpts(
    weights={
        "cumsum_freq": 1.0,
    },
    spectral_distance_type="L2",
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
