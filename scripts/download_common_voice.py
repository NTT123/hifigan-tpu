"""
Download and prepare infore dataset
"""
import os
from argparse import ArgumentParser
from pathlib import Path

import pooch
from pooch import Untar
from tqdm.cli import tqdm


def download_cv():
    files = pooch.retrieve(
        url="https://huggingface.co/datasets/ntt123/infore/resolve/main/cv-corpus-8.0-2022-01-19-vi.tar.gz",
        known_hash="38782df109852d3cbc6a7b788bfa3a745648c1886a4e81acd2a600b529a4fbe5",
        processor=Untar(),
        progressbar=True,
    )
    data_dir = Path(sorted(files)[0]).parent.parent
    return data_dir


parser = ArgumentParser()
parser.add_argument("--output-dir", type=Path, default="data")
args = parser.parse_args()


data_dir = download_cv()
wav_dir = data_dir / "clips"

for mp3_file in tqdm(wav_dir.glob("*.mp3")):
    out_dir = args.output_dir / "common_voice_spk"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_file = out_dir / (mp3_file.stem + ".wav")
    os.system(f"sox {mp3_file} -r 16k -b 16 -c 1 --norm=-3 {wav_file}")
