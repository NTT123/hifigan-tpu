"""TPU training script"""
import datetime
import os
import pickle
import random
import time
from pathlib import Path

import fire
import jax
import jax.numpy as jnp
import jax.tools.colab_tpu
import numpy as np
import opax
import pax
import tensorflow as tf

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
    grads = jax.lax.pmean(grads, axis_name="i")
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


def one_update_step(nets_optims, inputs):
    """update nets"""
    nets, optims = nets_optims
    generator, critics, mel_filter = nets
    cond, y = inputs
    mel = mel_filter(y)
    inputs = cond, y, mel
    optim_d, optim_g = optims
    vag = pax.value_and_grad(loss_fn, has_aux=True)
    (
        loss_gen_all,
        (generator, critics, optim_d, (loss_mel, loss_disc_all)),
    ), grads = vag(generator, (inputs, critics, optim_d, mel_filter))

    grads = jax.lax.pmean(grads, axis_name="i")

    generator, optim_g = opax.apply_gradients(generator, optim_g, grads)
    nets = (generator, critics, mel_filter)
    optims = (optim_d, optim_g)
    losses = (loss_gen_all, loss_mel, loss_disc_all)
    return (nets, optims), losses


def update_fn(nets, optims, inputs):
    """update fn"""
    inputs = jax.tree_map(lambda x: x.astype(jnp.float32), inputs)
    (nets, optims), losses = pax.scan(one_update_step, (nets, optims), inputs)
    losses = jax.tree_map(lambda x: x[-1], losses)
    return nets, optims, losses


def get_num_batch(data_dir, batch_size):
    files = sorted(Path(data_dir).glob("*.npz"))
    return len(files) // batch_size


def _device_put_sharded(sharded_tree, devices):
    leaves, treedef = jax.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return jax.device_put_sharded(
        [jax.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)], devices
    )


def double_buffer(ds, devices):
    """
    create a double buffer iterator
    """
    batch = None
    for next_batch in ds:
        assert next_batch is not None
        next_batch = _device_put_sharded(next_batch, devices)
        if batch is not None:
            yield batch
        batch = next_batch
    if batch is not None:
        yield batch


def load_npz_files(data_dir, test_size=10, split="train"):
    """return list of npz file in a directory"""
    files = sorted(Path(data_dir).glob("*.npz"))
    assert len(files) > 0, "Empty data directory"
    assert len(files) > test_size, "Empty test data size"
    if split == train:
        return files[test_size:]
    else:
        return files[:test_size]


def load_data(data_dir, config, devices, spu, pad):
    """return data iter"""
    files = load_npz_files(data_dir, split="train")
    batch = []
    cache = {}
    while True:
        random.shuffle(files)
        num_devices = len(devices)
        for f in files:
            if f in cache:
                mel, y = cache[f]
            else:
                dic = np.load(f)
                mel, y = dic["mel"], dic["y"]
                mel = np.pad(mel, [(pad, pad), (0, 0)], mode="reflect")
                cache[f] = mel, y
            num_frame = config.segment_size // config.hop_size
            start_frame = random.randint(0, mel.shape[0] - 1 - num_frame - pad * 2)
            end_frame = start_frame + num_frame + pad * 2
            mel = mel[start_frame:end_frame]
            start_idx = start_frame * config.hop_size
            end_idx = start_idx + config.segment_size
            y = y[start_idx:end_idx]
            batch.append((mel, y))
            if len(batch) == config.batch_size * spu:
                mel, y = zip(*batch)
                mel = np.array(mel).astype(np.float16)
                y = np.array(y).astype(np.float16)
                mel = mel.reshape((num_devices, spu, -1, *mel.shape[1:]))
                y = y.reshape((num_devices, spu, -1, *y.shape[1:]))
                yield mel, y
                batch = []


def save_ckpt(ckpt_dir, step, nets, optims):
    """save checkpoint to disk"""
    (generator, critics, _) = nets
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
        pickle.dump(dic, f)


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
    optim_g = optim_g.load_state_dict(dic["optim_g"])
    optims = (optim_d, optim_g)
    nets = (generator, critics, mel_filter)
    return step, nets, optims


@jax.jit
def fast_gen(generator, cond):
    """generate with padded input"""
    p = generator.compute_padding_values()[0]
    cond = jnp.pad(cond, [(0, 0), (p, p), (0, 0)], mode="reflect")
    return generator(cond)


def train(data_dir: str, log_dir: str = "logs", spu: int = 40):
    """train model...

    Arguments:
        data_dir: path to data directory
        spu: number of update steps per pmap update call
    """

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(str(Path(log_dir) / current_time))

    # TPU setup
    if "COLAB_TPU_ADDR" in os.environ:
        jax.tools.colab_tpu.setup_tpu()

    generator, critics, mel_filter = create_model(CONFIG)
    num_batch = get_num_batch(data_dir, CONFIG.batch_size)
    print(f"Data set size: {num_batch} batches")
    num_devices = jax.device_count()
    devices = jax.devices()
    print(f"{num_devices} devices: {devices}")

    def exp_decay(step):
        num_epoch = jnp.floor(step / 1000)
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
    input_pad = nets[0].compute_padding_values()[0]
    optims = (optim_d, optim_g)

    Path(CONFIG.ckpt_dir).mkdir(exist_ok=True, parents=True)
    last_step, nets, optims = load_ckpt(CONFIG.ckpt_dir, nets, optims)
    nets, optims = jax.device_put_replicated((nets, optims), devices)
    start = time.perf_counter()
    pmap_update_fn = jax.pmap(update_fn, axis_name="i", devices=devices)
    test_files = load_npz_files(data_dir, split="test")
    step = 0 if last_step == -1 else (last_step + spu)
    # log all ground-truth test audio clips
    with summary_writer.as_default(step=step):
        for f in test_files:
            dic = np.load(f)
            tf.summary.audio(
                f"gt/{f.stem}", dic["y"][None, :, None], CONFIG.sample_rate
            )

    for batch in double_buffer(
        load_data(data_dir, CONFIG, devices, spu, input_pad), devices
    ):
        nets, optims, (g_loss, mel_loss, d_loss) = pmap_update_fn(nets, optims, batch)

        if step % 50 == 0:
            end = time.perf_counter()
            dur = end - start
            start = end
            epoch = step // num_batch
            lr = optims[0][-1].learning_rate[0]
            print(
                f"step {step:07d}  epoch {epoch:05d}  lr {lr:.2e}  gen loss {g_loss[0]:.3f}"
                f"  mel loss {mel_loss[0]:.3f}  critic loss {d_loss[0]:.3f}  {dur:.2f}s"
            )

            with summary_writer.as_default(step=step):
                tf.summary.scalar("step", step)
                tf.summary.scalar("epoch", epoch)
                tf.summary.scalar("lr", lr)
                tf.summary.scalar("gen_loss", g_loss[0])
                tf.summary.scalar("mel_loss", mel_loss[0])
                tf.summary.scalar("critic_loss", d_loss[0])
                tf.summary.scalar("duration", dur)

        if step % 1000 == 0:
            with summary_writer.as_default(step=step):
                g = jax.tree_map(lambda x: x[0], nets[0].eval())
                g = jax.device_put(g, device=jax.devices("cpu")[0])
                for path in test_files:
                    dic = np.load(path)
                    yhat = fast_gen(g, dic["mel"][None].astype(jnp.float32))
                    yhat = jax.device_get(yhat)
                    tf.summary.audio(
                        f"gen/{path.stem}",
                        yhat[:, :, None],
                        CONFIG.sample_rate,
                    )

        if step % 10_000 == 0:
            nets_, optims_ = jax.device_get(
                jax.tree_map(lambda x: x[0], (nets, optims))
            )
            save_ckpt(Path(CONFIG.ckpt_dir), step, nets_, optims_)

        step += spu


if __name__ == "__main__":
    fire.Fire(train)
