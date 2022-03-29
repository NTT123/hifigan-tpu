"""
JAX DSP utility functions
"""

from functools import partial

import jax
import jax.numpy as jnp
import librosa
import pax
from jax.numpy import ndarray


def rolling_window(a: ndarray, window: int, hop_length: int):
    """return a stack of overlap subsequence of an array.
    ``return jnp.stack( [a[0:10], a[5:15], a[10:20],...], axis=0)``
    Source: https://github.com/google/jax/issues/3171
    Args:
      a (ndarray): input array of shape `[L, ...]`
      window (int): length of each subarray (window).
      hop_length (int): distance between neighbouring windows.
    """

    idx = (
        jnp.arange(window)[:, None]
        + jnp.arange((len(a) - window) // hop_length + 1)[None, :] * hop_length
    )
    return a[idx]


@partial(jax.jit, static_argnums=[1, 2, 3, 4])
def batched_stft(
    y: ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
):
    """Batched version of ``stft`` function.
    TN => FTN
    """

    assert len(y.shape) >= 2
    if window == "hann":
        fft_window = jnp.hanning(win_length + 1)[:-1]
    else:
        raise RuntimeError(f"'{window}' window function is not supported!")
    pad_len = (n_fft - win_length) // 2
    if pad_len > 0:
        fft_window = jnp.pad(fft_window, (pad_len, pad_len), mode="constant")
        win_length = n_fft

    p = (n_fft - hop_length) // 2
    y = jnp.pad(y, [(p, p), (0, 0)], mode="reflect")

    # jax does not support ``np.lib.stride_tricks.as_strided`` function
    # see https://github.com/google/jax/issues/3171 for comments.
    y_frames = rolling_window(y, n_fft, hop_length)
    fft_window = jnp.reshape(fft_window, (-1,) + (1,) * (len(y.shape)))
    y_frames = y_frames * fft_window
    stft_matrix = jnp.fft.rfft(y_frames, axis=0)
    return stft_matrix


class MelFilter(pax.Module):
    """Convert waveform to mel spectrogram."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        window_length: int,
        hop_length: int,
        n_mels: int,
        fmin=0.0,
        fmax=8000,
        mel_min=1e-5,
    ):
        super().__init__()
        self.melfb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.n_fft = n_fft
        self.window_length = window_length
        self.hop_length = hop_length
        self.mel_min = mel_min

    def __call__(self, y: ndarray) -> ndarray:
        hop_length = self.hop_length
        window_length = self.window_length
        assert len(y.shape) == 2
        spec = batched_stft(y.T, self.n_fft, hop_length, window_length, "hann")
        mag = jnp.sqrt(jnp.square(spec.real) + jnp.square(spec.imag) + 1e-9)
        mel = jnp.einsum("ms,sfn->nfm", self.melfb, mag)
        cond = jnp.log(jnp.clip(mel, a_min=self.mel_min, a_max=None))
        return cond
