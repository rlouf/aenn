import aesara.tensor as at

from aenn.nn.activation import ReLu, relu


def test_relu():

    X = at.vector("X")
    res = relu(X)
    assert isinstance(res.owner.op, ReLu)
