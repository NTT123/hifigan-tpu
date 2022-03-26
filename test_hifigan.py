import jax
import jax.numpy as jnp
import pax

from hifigan import SpectralNormConv, WeightNormConv


def test_weight_norm():
    """make sure the weight is normalized per output channel."""
    conv = pax.Conv1D(10, 20, 9, 3, 2)
    net = WeightNormConv(conv)
    net = net.replace(g=net.g * 0.5)
    x = jnp.ones((3, 128, 10))
    y1 = net(x)

    # make sure remove_weight_norm works!
    net = net.remove_weight_norm()
    assert net.g is None
    y2 = net(x)
    assert jnp.all(y2 == y1).item()


def test_spectral_norm():
    """test spectral norm (power iterator) computation"""
    conv = pax.Conv1D(10, 20, 9, 3, 2, w_init=jax.nn.initializers.normal(0.01))
    assert conv.weight.shape[-1] == 20
    w = conv.weight.reshape([-1, 20])
    spectral_norm = jnp.max(jnp.sqrt(jnp.real(jnp.linalg.eigvals(jnp.matmul(w.T, w)))))
    net = SpectralNormConv(conv)
    x = jnp.ones((3, 128, 10))
    for _ in range(100):
        net, _ = pax.purecall(net, x)
    assert jnp.abs(net.sigma - spectral_norm).item() < 1e-3
