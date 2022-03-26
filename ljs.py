"""
Download LJSpeech dataset
"""

from pathlib import Path

import pooch
from pooch import Untar


def download_ljs_dataset():
    """
    Download ljs dataset and save it to disk
    """

    file_paths = pooch.retrieve(
        url="https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
        known_hash="md5:c4763be9595ddfa79c2fc6eaeb3b6c8e",
        processor=Untar(),
        progressbar=True,
    )
    readme_file = Path(sorted(file_paths)[0])
    assert readme_file.name == "README"
    return readme_file.parent


data_dir = download_ljs_dataset()
wav_dir = data_dir / "wavs"
print(f"Downloaded data is located at {wav_dir}")
