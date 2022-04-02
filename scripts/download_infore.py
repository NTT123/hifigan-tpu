"""
Download and prepare infore dataset
"""
import os
from argparse import ArgumentParser
from pathlib import Path

import pooch
from pooch import Unzip
from tqdm.cli import tqdm


def download_infore_data():
    files = pooch.retrieve(
        url="https://huggingface.co/datasets/ntt123/infore/resolve/main/infore_16k.zip",
        known_hash="0c9b2fd6962fd6706fa9673f94a9f1ba534edf34691247ae2be0ff490870ccd7",
        processor=Unzip(),
        progressbar=True,
    )
    data_dir = Path(sorted(files)[0]).parent
    return data_dir


parser = ArgumentParser()
parser.add_argument("--output-dir", type=Path, default="data")
args = parser.parse_args()

data_dir = download_infore_data()
wav_files = sorted(data_dir.glob("*.wav"))

out_dir = args.output_dir / "infore_spk"
out_dir.mkdir(parents=True, exist_ok=True)

for wav_file in tqdm(wav_files):
    new_wav_file = out_dir / wav_file.name
    os.system(f"sox {wav_file} -r 16k -b 16 -c 1 --norm=-3 {new_wav_file}")
