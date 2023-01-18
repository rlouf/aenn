from typing import Optional

import aesara.tensor as at
from aesara.compile.builders import OpFromGraph
from aesara.tensor.var import TensorVariable


class Parameter(TensorVariable):
    """DefaultValue"""


class DenseLayer(OpFromGraph):
    """Represents a dense layer Op."""

    def __init__(self, units, inputs, outputs):
        self.units = units
        super().__init__(inputs, outputs)


class Dense:
    def __init__(
        self,
        units: int,
        W: TensorVariable = Parameter,
        b: Optional[TensorVariable] = Parameter,
    ):
        """Initialize the dense layer.

        We can initialize a dense layer by specifying the number of hidden
        units. In this case, building the `Op` will initialize `W` and `b` with
        the corresponding shape.

        >>> import aesara.tensor as at
        >>> from aenn.nn.dense import Dense
        >>> x = at.matrix('x')
        >>> x = Dense(50)(x)

        We can also specify the weight and bias matrix/vector. This can be
        useful to define Bayesian Neural Networks, for instance:

        >>> srng = at.random.RandomStream(0)
        >>> W = srng.normal(0, 1, size=(x.shape[1], 50))
        >>> x = Dense(50, W)(x)

        There is a bit of repetition here, but I am afraid this is unavoidable.
        I thought about using `multipledispatch` and register an `__init__`
        method for an `int` input and another for a `TensorVariable`, but there
        is the case where we'd like to pass a scalar tensor for `unit` to be
        able to change the layer size after compilation:

        >>> units = at.iscalar("units")
        >>> x = Dense(units)(x)

        We can choose to remove the layers' bias by setting `b` to `None`:

        >>> x = Dense(50, b=None)(x)

        The number of units passed to the `__init__` function is only used
        during initialization. Since layers can be affected by rewrites and see
        their shape change, we let `W` and `b` with undefined shapes (but fixed
        number of dimensions) and update this initialization number instead.

        """
        self.units = units
        self.W = W
        self.b = b

    def __call__(self, x: TensorVariable):

        if self.W == Parameter:
            self.W = at.matrix()

        if self.b is None:
            output = at.dot(x, self.W)
            dense = DenseLayer(self.units, [x, self.W], [output])
            return dense(x, self.W)

        if self.b == Parameter:
            self.b = at.vector()

        output = at.dot(x, self.W) + self.b
        dense = DenseLayer(self.units, [x, self.W, self.b], [output])
        return dense(x, self.W, self.b)
