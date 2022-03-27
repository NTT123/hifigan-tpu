import pickle
from argparse import ArgumentParser
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io.wavfile import write

import config
from hifigan import Generator

parser = ArgumentParser()
parser.add_argument("--model", type=Path, required=True, help="Path to model file")
parser.add_argument("--mel", type=Path, required=True, help="Path to mel file")
parser.add_argument("--wav", type=Path, required=True, help="Path to output wav file")
args = parser.parse_args()


g = Generator(
    config.num_mels,
    config.resblock_kernel_sizes,
    config.upsample_rates,
    config.upsample_kernel_sizes,
    config.upsample_initial_channel,
    config.resblock_kind,
    config.resblock_dilation_sizes,
)

with open(args.model, "rb") as f:
    dic = pickle.load(f)

g.load_state_dict(dic["generator"])
g = jax.device_put(g, device=jax.devices("cpu")[0])
g = g.eval()
mel = np.load(args.mel)
wav = g(mel)

wav = wav * 2**15
wav = jnp.clip(wav, a_min=-(2**15), a_max=2**15 - 1)
wav = wav.astype(jnp.int16)
wav = jax.device_get(wav)


write(args.wav, config.sample_rate, wav)
print("Done!")
