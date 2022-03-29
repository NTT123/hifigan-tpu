"""
Prepare data for normal training
"""
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import librosa
import numpy as np
from tqdm.cli import tqdm

import config as CONFIG

parser = ArgumentParser()
parser.add_argument("--wav-dir", type=Path, required=True)

args = parser.parse_args()

wav_files = sorted(args.wav_dir.glob("*.wav"), key=os.path.getsize)
random.Random(42).shuffle(wav_files)
data = []
for path in tqdm(wav_files):
    y, sr = librosa.load(path, sr=CONFIG.sample_rate)
    assert sr == CONFIG.sample_rate
    fn = path.with_suffix(".npz")
    np.savez(fn, y=y)
