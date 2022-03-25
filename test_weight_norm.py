import jax.numpy as jnp
import pax

from hifigan import WeightNormConv


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
