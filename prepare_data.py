import os
import random
from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import librosa
import numpy as np
from tqdm.cli import tqdm

import config as CONFIG
from dsp import MelFilter

parser = ArgumentParser()
parser.add_argument("--wav-dir", type=Path, required=True)

args = parser.parse_args()


mel_filter = MelFilter(
    CONFIG.sample_rate,
    CONFIG.n_fft,
    CONFIG.win_size,
    CONFIG.hop_size,
    CONFIG.num_mels,
    CONFIG.fmin,
    CONFIG.fmax,
)
mel_fn = jax.jit(lambda x: mel_filter(x[None])[0].astype(jnp.float16))

# save mel to disk
wav_files = sorted(args.wav_dir.glob("*.wav"), key=os.path.getsize)
L = len(librosa.load(wav_files[-1], sr=None)[0])
random.Random(42).shuffle(wav_files)
data = []
for path in tqdm(wav_files):
    y, sr = librosa.load(path, sr=None)
    l = len(y)
    y = np.pad(y, (0, L - len(y)))
    mel = mel_fn(jax.device_put(y))
    mel = jax.device_get(mel)
    assert sr == CONFIG.sample_rate
    y, mel = y[:l], mel[: (l // CONFIG.hop_size)]
    fn = path.with_suffix(".npz")
    np.savez(fn, y=y, mel=mel)
