"""GPU training script"""
import pickle
import random
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
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
    """Multiple Critics"""

    def __init__(self, mpd, msd):
        super().__init__()
        self.mpd = mpd
        self.msd = msd


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
    return g, Critics(mpd=mpd, msd=msd), mel_filter


@pax.pure
def critic_loss_fn(critics: Critics, inputs):
    """critic loss"""
    y, y_g_hat = inputs

    # MPD
    y_df_hat_r, y_df_hat_g, _, _ = critics.mpd(y, y_g_hat)
    loss_disc_f, losses_disc_f_r, losses_disc_f_g = critic_loss(y_df_hat_r, y_df_hat_g)

    # MSD
    y_ds_hat_r, y_ds_hat_g, _, _ = critics.msd(y, y_g_hat)
    loss_disc_s, losses_disc_s_r, losses_disc_s_g = critic_loss(y_ds_hat_r, y_ds_hat_g)
    loss_disc_all = loss_disc_s + loss_disc_f
    return loss_disc_all, critics


def update_critic(nets, optims, inputs):
    """update critic"""
    (loss_disc_all, nets), grads = pax.value_and_grad(critic_loss_fn, has_aux=True)(
        nets, inputs
    )
    nets, optims = opax.apply_gradients(nets, optims, grads)
    return nets, optims, loss_disc_all


def l1_loss(a, b):
    """l1 loss function"""
    return jnp.mean(jnp.abs(a - b))


def loss_fn(generator, inputs):
    """main loss function"""
    (x, y, y_mel), critics, optim_d, mel_filter = inputs
    y_g_hat = generator(x)
    y_g_hat_mel = mel_filter(y_g_hat)
    y, y_g_hat = jax.tree_map(lambda t: t[..., None], (y, y_g_hat))
    critics, optim_d, loss_disc_all = jax.lax.stop_gradient(
        update_critic(critics, optim_d, (y, y_g_hat))
    )

    loss_mel = l1_loss(y_mel, y_g_hat_mel) * 45
    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = critics.mpd(y, y_g_hat)
    # did not update spectral norm here
    y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = critics.msd.eval()(y, y_g_hat)
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


def get_num_batch(data_dir, batch_size):
    files = sorted(Path(data_dir).glob("*.npz"))
    return len(files) // batch_size


def load_data(data_dir, config):
    """return data iter"""
    files = sorted(Path(data_dir).glob("*.npz"))
    assert len(files) > 0, "Empty data directory"
    batch = []
    random.shuffle(files)
    for f in files:
        dic = np.load(f)
        mel, y = dic["mel"], dic["y"]
        num_frame = config.segment_size // config.hop_size
        start_frame = random.randint(0, mel.shape[0] - 1 - num_frame)
        end_frame = start_frame + num_frame
        mel = mel[start_frame:end_frame]
        start_idx = start_frame * config.hop_size
        end_idx = start_idx + config.segment_size
        y = y[start_idx:end_idx]
        batch.append((mel, y))
        if len(batch) == config.batch_size:
            mel, y = zip(*batch)
            mel = np.array(mel).astype(np.float32)
            y = np.array(y)
            yield mel, y, mel
            batch = []


def save_ckpt(ckpt_dir, step, nets, optims):
    """save checkpoint to disk"""
    (generator, critics, mel_filter) = nets
    (optim_d, optim_g) = optims
    dic = {
        "step": step,
        "generator": generator.state_dict(),
        "critics": critics.state_dict(),
        "optim_d": optim_d.state_dict(),
        "optim_g": optim_g.state_dict(),
    }
    file = ckpt_dir / f"hifigan_{step:07d}.ckpt"
    with open(file, "wb") as f:
        pickle.dump(f, dic)


def load_ckpt(ckpt_dir, nets, optims):
    """load latest checkpoint from disk"""
    ckpts = sorted(Path(ckpt_dir).glob("hifigan_*.ckpt"))
    if len(ckpts) == 0:
        return -1, nets, optims

    print("Loading checkpoint at ", ckpts[-1])
    (generator, critics, mel_filter) = nets
    (optim_d, optim_g) = optims
    with open(ckpts[-1], "rb") as f:
        dic = pickle.load(f)
    step = dic["step"]
    generator = generator.load_state_dict(dic["generator"])
    critics = critics.load_state_dict(dic["critics"])
    optim_d = optim_d.load_state_dict(dic["optim_d"])
    optim_g = optim_d.load_state_dict(dic["optim_g"])
    optims = (optim_d, optim_g)
    nets = (generator, critics, mel_filter)
    return step, nets, optims


def train(data_dir):
    """train model..."""
    generator, critics, mel_filter = create_model(CONFIG)

    def exp_decay(step):
        num_epoch = jnp.floor(step / get_num_batch(data_dir, CONFIG.batch_size))
        scale = jnp.power(CONFIG.lr_decay, num_epoch)
        return scale * CONFIG.learning_rate

    optim = opax.chain(
        opax.scale_by_adam(b1=CONFIG.adam_b1, b2=CONFIG.adam_b2),
        opax.add_decayed_weights(1e-2),
        opax.scale_by_schedule(exp_decay),
    )
    optim_g = optim.init(generator.parameters())
    optim_d = optim.init((critics.parameters()))

    nets = (generator, critics, mel_filter)
    optims = (optim_d, optim_g)

    Path(CONFIG.ckpt_dir).mkdir(exist_ok=True, parents=True)
    step, net, optims = load_ckpt(CONFIG.ckpt_dir, nets, optims)

    for epoch in range(10000):
        for batch in load_data(data_dir, CONFIG):
            step += 1
            nets, optims, (gen_loss, mel_loss, critic_loss) = update_fn(
                nets, optims, batch
            )

            if step % 100 == 0:
                print(
                    f"step {step:07d}  epoch {epoch:05d}  gen loss {gen_loss:.3f}  mel loss {mel_loss:.3f}  critic loss {critic_loss:.3f}"
                )

        if epoch % 20 == 0:
            save_ckpt(CONFIG.ckpt_dir, step, nets, optims)


if __name__ == "__main__":
    fire.Fire(train)
