import aesara.tensor as at
from aesara.compile.builders import OpFromGraph


class ReLu(OpFromGraph):
    """Represents a ReLu activation"""


def relu(x):
    r"""Rectified linear unit activation.

    Computes the element-wise function:

    .. math::

        \operatorname{relu}(x) = \max(x, 0)

    """
    output = at.where(x > 0, x, 0)
    relu = ReLu([x], [output])
    return relu(x)


class LeakyReLu(OpFromGraph):
    """Represents a leaky ReLu activation"""


def leaky_relu(x, negative_slope=1e-2):
    r"""Leaky rectified linear unit activation.

    Computes the element-wise function:

    .. math::
        \mathrm{leaky\_relu}(x) = \begin{cases}
        x, & x \ge 0\\
        \alpha x, & x < 0
        \end{cases}

    where :math:`\alpha` represents :code:`negative_slope`.

    """
    output = at.where(x >= 0, x, negative_slope * x)
    leaky_relu = LeakyReLu([x], [output])
    return leaky_relu(x)
