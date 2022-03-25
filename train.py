"""Training script"""
import random
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import librosa
import numpy as np
import opax
import pax

import config as CONFIG
from dsp import MelFilter
from hifigan import (
    Generator,
    MultiPeriodCritic,
    MultiScaleCritic,
    critic_loss,
    feature_loss,
    generator_loss,
)


class Critics(pax.Module):
    """Critics"""

    def __init__(self, *s):
        super().__init__()
        self.s = s

    def __iter__(self):
        return iter(self.s)


def create_model(config):
    """return a new model"""
    g = Generator(
        config.num_mels,
        config.resblock_kernel_sizes,
        config.upsample_rates,
        config.upsample_kernel_sizes,
        config.upsample_initial_channel,
        config.resblock_kind,
        config.resblock_dilation_sizes,
    )
    mpd = MultiPeriodCritic()
    msd = MultiScaleCritic()
    mel_filter = MelFilter(
        config.sample_rate,
        config.n_fft,
        config.win_size,
        config.hop_size,
        config.num_mels,
        config.fmin,
        config.fmax,
    )
    return g, Critics(mpd, msd), mel_filter


@pax.pure
def critic_loss_fn(nets, inputs):
    """critic loss"""
    mpd, msd = nets
    y, y_g_hat = inputs

    # MPD
    y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat)
    loss_disc_f, losses_disc_f_r, losses_disc_f_g = critic_loss(y_df_hat_r, y_df_hat_g)

    # MSD
    y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat)
    loss_disc_s, losses_disc_s_r, losses_disc_s_g = critic_loss(y_ds_hat_r, y_ds_hat_g)
    loss_disc_all = loss_disc_s + loss_disc_f
    return loss_disc_all, Critics(mpd, msd)


def update_critic(nets, optims, inputs):
    """update critic"""
    (loss_disc_all, nets), grads = pax.value_and_grad(critic_loss_fn, has_aux=True)(
        nets, inputs
    )
    nets, optims = opax.apply_gradients(nets, optims, grads)
    return nets, optims, loss_disc_all


def l1_loss(a, b):
    return jnp.mean(jnp.abs(a - b))


def loss_fn(generator, inputs):
    """main loss function"""
    (x, y, y_mel), critics, optim_d, mel_filter = inputs
    y_g_hat = generator(x)
    y_g_hat_mel = mel_filter(y_g_hat)
    critics, optim_d, loss_disc_all = jax.lax.stop_gradient(
        update_critic(critics, optim_d, (y[..., None], y_g_hat[..., None]))
    )

    loss_mel = l1_loss(y_mel, y_g_hat_mel) * 45
    mpd, msd = critics
    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd.eval()(
        y[..., None], y_g_hat[..., None]
    )
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd.eval()(
        y[..., None], y_g_hat[..., None]
    )
    loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
    loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
    loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

    return loss_gen_all, (generator, critics, optim_d, (loss_mel, loss_disc_all))


@jax.jit
def update_fn(nets, optims, inputs):
    """update nets"""
    generator, critics, mel_filter = nets
    mel = mel_filter(inputs)
    inputs = (mel, inputs, mel)
    optim_d, optim_g = optims
    vag = pax.value_and_grad(loss_fn, has_aux=True)
    (
        loss_gen_all,
        (generator, critics, optim_d, (loss_mel, loss_disc_all)),
    ), grads = vag(generator, (inputs, critics, optim_d, mel_filter))

    generator, optim_g = opax.apply_gradients(generator, optim_g, grads)
    return (
        (generator, critics, mel_filter),
        (optim_d, optim_g),
        (loss_gen_all, loss_mel, loss_disc_all),
    )


def data_iter(batch_size, config):
    """return data"""
    files = sorted(Path(config.wav_dir).glob("*.wav"))
    assert len(files) > 0, "Empty wav directory"
    rand = random.Random(42)
    rand.shuffle(files)
    batch = []
    while True:
        rand.shuffle(files)
        for f in files:
            y, sr = librosa.load(f, sr=None)
            assert sr == config.sample_rate
            i = rand.randint(0, len(y) - config.segment_size)
            s = y[i : i + config.segment_size]
            batch.append(s)
            if len(batch) == batch_size:
                yield np.array(batch)
                batch = []


def train(
    batch_size: int,
):
    """Train..."""
    generator, critics, mel_filter = create_model(CONFIG)

    def exp_decay(step):
        lr = jnp.power(CONFIG.lr_decay, step)
        return lr * CONFIG.learning_rate

    optim = opax.chain(
        opax.scale_by_adam(b1=CONFIG.adam_b1, b2=CONFIG.adam_b2),
        opax.add_decayed_weights(1e-2),
        opax.scale_by_schedule(exp_decay),
    )
    optim_g = optim.init(generator.parameters())
    optim_d = optim.init((critics.parameters()))

    nets = (generator, critics, mel_filter)
    optims = (optim_d, optim_g)

    step = 0

    for batch in data_iter(batch_size, CONFIG):
        step += 1
        nets, optims, (gen_loss, mel_loss, critic_loss) = update_fn(nets, optims, batch)

        if step % 100 == 0:
            print(
                f"step {step:07d}  gen loss {gen_loss:.3f}  mel loss {mel_loss:.3f}  critic loss {critic_loss:.3f}"
            )


if __name__ == "__main__":
    fire.Fire(train)
