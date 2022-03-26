"""
HiFi-GAN model: generator and critics.
"""


import jax
import jax.numpy as jnp
import pax

LRELU_SLOPE = 0.1


# Source: https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/spectral_norm.py
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
def _l2_normalize(x, axis=None, eps=1e-12):
    """Normalizes along dimension `axis` using an L2 norm.
    This specialized function exists for numerical stability reasons.
    Args:
      x: An input ndarray.
      axis: Dimension along which to normalize, e.g. `1` to separately normalize
        vectors in a batch. Passing `None` views `t` as a flattened vector when
        calculating the norm (equivalent to Frobenius norm).
      eps: Epsilon to avoid dividing by zero.
    Returns:
      An array of the same shape as 'x' L2-normalized along 'axis'.
    """
    return x * jax.lax.rsqrt((x * x).sum(axis=axis, keepdims=True) + eps)


class WeightNormConv(pax.Module):
    """Weight norm normalized convolution"""

    def __init__(self, conv: pax.Conv1D):
        super().__init__()
        self.conv = conv
        if isinstance(conv, pax.Conv1D):
            self.g = jnp.ones((1, 1, conv.out_features))
        elif isinstance(conv, pax.Conv1DTranspose):
            self.g = jnp.ones((1, conv.out_features, 1))

    parameters = pax.parameters_method("g")

    def get_weight(self):
        """compute the normalized weight"""
        assert self.g is not None, "Missing parameter `g`."
        if isinstance(self.conv, pax.Conv1D):
            weight = _l2_normalize(self.conv.weight, (0, 1))
        elif isinstance(self.conv, pax.Conv1DTranspose):
            weight = _l2_normalize(self.conv.weight, (0, 2))
        assert weight.shape == self.conv.weight.shape
        weight = self.g * weight
        return weight

    def __call__(self, x):
        """compute conv"""
        if self.g is None:
            return self.conv(x)

        return self.conv.replace(weight=self.get_weight())(x)

    def remove_weight_norm(self):
        """
        remove g parameter for better performance
        """
        conv = self.conv.replace(weight=self.get_weight())
        return self.replace(conv=conv, g=None)


class SpectralNormConv(pax.Module):
    """Spectral norm normalized convolution"""

    def __init__(
        self,
        conv: pax.Conv1D,
        eps: float = 1e-4,
        n_steps: int = 1,
    ):
        super().__init__()
        self.conv = conv
        self.eps = eps
        self.n_steps = n_steps
        self.u0 = jax.random.normal(pax.next_rng_key(), (1, conv.out_features))
        self.sigma = jnp.ones(())

    def get_weight(self):
        """get normalized weight"""
        weight = self.conv.weight

        # Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
        # Licensed under the Apache License, Version 2.0
        value = jnp.reshape(weight, [-1, weight.shape[-1]])
        if self.training:
            u0 = self.u0
            # Power iteration for the weight's singular value.
            for _ in range(self.n_steps):
                v0 = _l2_normalize(
                    jnp.matmul(u0, value.transpose([1, 0])), eps=self.eps
                )
                u0 = _l2_normalize(jnp.matmul(v0, value), eps=self.eps)

            u0 = jax.lax.stop_gradient(u0)
            v0 = jax.lax.stop_gradient(v0)
            sigma = jnp.matmul(jnp.matmul(v0, value), jnp.transpose(u0))[0, 0]
            self.u0 = u0
            self.sigma = sigma
        else:
            sigma = self.sigma

        value /= sigma
        value_bar = value.reshape(weight.shape)

        return value_bar

    def __call__(self, x):
        return self.conv.replace(weight=self.get_weight())(x)


def normalized_conv(
    input,
    output,
    kernel_size,
    stride,
    dilation=1,
    padding="SAME",
    group=1,
    spectral_norm=False,
):
    """return a 'normalized' conv"""
    mod = pax.Conv1D(
        input,
        output,
        kernel_size,
        stride=stride,
        rate=dilation,
        padding=padding,
        feature_group_count=group,
        w_init=jax.nn.initializers.normal(0.01),
    )

    if spectral_norm:
        return SpectralNormConv(mod)

    return WeightNormConv(mod)


def conv_transpose(in_channel, out_channel, kernel_size, upsample_factor):
    """return a conv transpose"""
    return WeightNormConv(
        pax.Conv1DTranspose(
            in_channel,
            out_channel,
            kernel_size,
            upsample_factor,
            padding="SAME",
            w_init=jax.nn.initializers.normal(0.01),
        )
    )


class ResBlock1(pax.Module):
    """ResBlock1 module"""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = [
            normalized_conv(channels, channels, kernel_size, 1, dilation[0]),
            normalized_conv(channels, channels, kernel_size, 1, dilation[1]),
            normalized_conv(channels, channels, kernel_size, 1, dilation[2]),
        ]
        self.convs2 = [
            normalized_conv(channels, channels, kernel_size, 1, 1),
            normalized_conv(channels, channels, kernel_size, 1, 1),
            normalized_conv(channels, channels, kernel_size, 1, 1),
        ]

    def __call__(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = jax.nn.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = jax.nn.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(pax.Module):
    """ResBlock2 module"""

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = [
            normalized_conv(channels, channels, kernel_size, 1, dilation[0]),
            normalized_conv(channels, channels, kernel_size, 1, dilation[1]),
        ]

    def __call__(self, x):
        for c in self.convs:
            xt = jax.nn.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class Generator(pax.Module):
    """HiFi-GAN Generator"""

    def __init__(
        self,
        mel_dim,
        resblock_kernel_sizes,
        upsample_rates,
        upsample_kernel_sizes,
        upsample_initial_channel,
        resblock_kind,
        resblock_dilation_sizes,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = normalized_conv(mel_dim, upsample_initial_channel, 7, 1, 1)
        create_resblock = ResBlock1 if resblock_kind == "1" else ResBlock2

        self.ups = []
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_channel = upsample_initial_channel // (2**i)
            self.ups.append(conv_transpose(in_channel, in_channel // 2, k, u))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**i) // 2
            for (k, d) in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(create_resblock(ch, k, d))

        self.conv_post = normalized_conv(ch, 1, 7, 1, 1)

    def __call__(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # 0.01 is pytorch leaky value slope,
        # this is not needed as jax uses the same value.
        x = jax.nn.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        x = jnp.tanh(x)
        x = jnp.squeeze(x, axis=-1)
        return x


class CriticP(pax.Module):
    """HiFi-GAN CriticP"""

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = [
            normalized_conv(1, 32, kernel_size, stride),
            normalized_conv(32, 128, kernel_size, stride),
            normalized_conv(128, 512, kernel_size, stride),
            normalized_conv(512, 1024, kernel_size, stride),
            normalized_conv(1024, 1024, kernel_size, stride),
        ]
        self.conv_post = normalized_conv(1024, 1, 3, 1, 1)

    def __call__(self, x: jnp.ndarray):
        fmap = []
        b, t, c = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, [(0, 0), (0, n_pad), (0, 0)])
            t = t + n_pad

        x = jnp.reshape(x, (b, t // self.period, self.period, c))
        x = jnp.swapaxes(x, 1, 2)
        x = jnp.reshape(x, (b * self.period, t // self.period, c))

        for conv in self.convs:
            x = conv(x)
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = jnp.reshape(x, (b, -1))
        return x, fmap


class MultiPeriodCritic(pax.Module):
    """Multi Period Critic"""

    def __init__(self):
        super().__init__()
        self.critics = [CriticP(2), CriticP(3), CriticP(5), CriticP(7), CriticP(11)]

    def __call__(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.critics:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class CriticS(pax.Module):
    """Scale Critic"""

    def __init__(self, spectral_norm=False):
        super().__init__()
        self.convs = [
            normalized_conv(1, 128, 15, 1, spectral_norm=spectral_norm),
            normalized_conv(128, 128, 41, 2, group=4, spectral_norm=spectral_norm),
            normalized_conv(128, 256, 41, 2, group=16, spectral_norm=spectral_norm),
            normalized_conv(256, 512, 41, 4, group=16, spectral_norm=spectral_norm),
            normalized_conv(512, 1024, 41, 4, group=16, spectral_norm=spectral_norm),
            normalized_conv(1024, 1024, 41, 1, group=16, spectral_norm=spectral_norm),
            normalized_conv(1024, 1024, 5, 1, spectral_norm=spectral_norm),
        ]
        self.conv_post = normalized_conv(1024, 1, 3, 1, spectral_norm=spectral_norm)

    def __call__(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = jax.nn.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = jnp.reshape(x, (x.shape[0], -1))
        return x, fmap


class MultiScaleCritic(pax.Module):
    """Multi Scale Critic"""

    def __init__(self):
        super().__init__()
        self.critics = [
            CriticS(spectral_norm=True),
            CriticS(),
            CriticS(),
        ]
        self.meanpools = [
            lambda x: x,
            lambda x: pax.avg_pool(x, 4, 2, "SAME", -1),
            lambda x: pax.avg_pool(x, 4, 2, "SAME", -1),
        ]

    def __call__(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.critics):
            y = self.meanpools[i](y)
            y_hat = self.meanpools[i](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    """feature loss"""
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += jnp.mean(jnp.abs(rl - gl))

    return loss * 2


def critic_loss(disc_real_outputs, disc_generated_outputs):
    """critic loss"""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = jnp.mean((1 - dr) ** 2)
        g_loss = jnp.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss)
        g_losses.append(g_loss)

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """generator loss"""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = jnp.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
