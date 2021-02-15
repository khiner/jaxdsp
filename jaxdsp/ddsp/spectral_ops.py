import numpy as np
import jax.numpy as jnp
import jax_spectral

LD_RANGE = 120.0  # dB

def stft(audio, sample_rate=16000, frame_size=2048, overlap=0.75, pad_end=True):
    assert frame_size * overlap % 2.0 == 0.0

    # Remove channel dim if present.
    if len(audio.shape) == 3:
        audio = jnp.squeeze(audio, axis=-1)

    return jax_spectral.spectral.stft(
        audio,
        # sample_rate,
        nperseg=int(frame_size),
        noverlap=int(overlap),
#        nfft=int(frame_size),
        padded=pad_end,
    )


def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
    return jnp.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end)[2])
