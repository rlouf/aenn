import aesara.tensor as at

from aenn.nn.dense import Dense, DenseLayer


def test_dense_shape_init():
    X = at.matrix("X")
    res = Dense(50)(X)

    W_shape = res.owner.inputs[1].type.shape
    b_shape = res.owner.inputs[2].type.shape

    assert len(W_shape) == 2
    assert len(b_shape) == 1
    assert isinstance(res.owner.op, DenseLayer)


def test_dense_init_no_bias():
    X = at.matrix("X")
    res = Dense(50, b=None)(X)
    assert len(res.owner.inputs) == 2


def test_dense_init_with_weights():
    X = at.matrix("X")
    W = at.matrix("W")
    b = at.vector("b")

    X = at.matrix("X")
    res = Dense(50, W, b)(X)

    assert res.owner.inputs[1] == W
    assert res.owner.inputs[2] == b
