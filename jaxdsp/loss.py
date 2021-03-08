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


# Based on https://github.com/magenta/ddsp/blob/master/ddsp/losses.py#L132
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
    ):
        """Constructor, set loss weights of various components.
        Args:
        fft_sizes: Compare spectrograms at each of this list of fft sizes. Each
            spectrogram has a time-frequency resolution trade-off based on fft size,
            so comparing multiple scales allows multiple resolutions.
        loss_type: One of 'L1', 'L2', or 'COSINE'.
        mag_weight: Weight to compare linear magnitudes of spectrograms. Core
            audio similarity loss. More sensitive to peak magnitudes than log
            magnitudes.
        delta_time_weight: Weight to compare the first finite difference of
            spectrograms in time. Emphasizes changes of magnitude in time, such as
            at transients.
        delta_freq_weight: Weight to compare the first finite difference of
            spectrograms in frequency. Emphasizes changes of magnitude in frequency,
            such as at the boundaries of a stack of harmonics.
        cumsum_freq_weight: Weight to compare the cumulative sum of spectrograms
            across frequency for each slice in time. Similar to a 1-D Wasserstein
            loss, this hopefully provides a non-vanishing gradient to push two
            non-overlapping sinusoids towards eachother.
        logmag_weight: Weight to compare log magnitudes of spectrograms. Core
            audio similarity loss. More sensitive to quiet magnitudes than linear
            magnitudes.
        """
        self.fft_sizes = fft_sizes
        self.distance_type = distance_type
        self.distance_fn = functools.partial(distance, kind=distance_type)
        self.mag_weight = mag_weight
        self.delta_time_weight = delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.cumsum_freq_weight = cumsum_freq_weight
        self.logmag_weight = logmag_weight
        self.spectrogram_fns = [
            functools.partial(magnitute_spectrogram, size=size)
            for size in self.fft_sizes
        ]


# Based on https://github.com/magenta/ddsp/blob/master/ddsp/losses.py#L132
def multi_scale_spectral(X, Y, opts):
    loss = 0.0
    distance_fn = opts.distance_fn
    for spectrogram_fn in opts.spectrogram_fns:
        X_mag = spectrogram_fn(X)
        Y_mag = spectrogram_fn(Y)

        if opts.mag_weight > 0:
            loss += opts.mag_weight * distance_fn(X_mag, Y_mag)
        if opts.delta_time_weight > 0:
            loss += opts.delta_time_weight * distance_fn(
                jnp.diff(X_mag, axis=1), jnp.diff(Y_mag, axis=1)
            )
        if opts.delta_freq_weight > 0:
            loss += opts.delta_freq_weight * distance_fn(
                jnp.diff(X_mag, axis=2), jnp.diff(Y_mag, axis=2)
            )
        if opts.cumsum_freq_weight > 0:
            loss += opts.cumsum_freq_weight * distance_fn(
                jnp.cumsum(X_mag, axis=2), jnp.cumsum(Y_mag, axis=2)
            )
        if opts.logmag_weight > 0:
            loss += opts.logmag_weight * distance_fn(safe_log(X_mag), safe_log(Y_mag))

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
