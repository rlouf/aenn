import aesara.tensor as at
from aesara.graph.features import ReplaceValidate
from aesara.graph.fg import FunctionGraph
from aesara.graph.rewriting.basic import (
    EquilibriumGraphRewriter,
    GraphRewriter,
    node_rewriter,
)
from aesara.graph.rewriting.utils import rewrite_graph

from aenn.nn.activation import ReLu, leaky_relu, relu
from aenn.nn.dense import Dense, DenseLayer


def test_mlp_swap_relu():
    """Make sure that we can swap activation functions."""
    x = at.matrix("X")
    W = at.matrix("W")

    x = Dense(100, W)(x)
    x = relu(x)
    x = Dense(10, b=None)(x)

    @node_rewriter([ReLu])
    def replace_relu(fgraph, node):
        if isinstance(node.op, ReLu):
            x = node.inputs[0]
            out = leaky_relu(x)
            return out.owner.outputs

    relu_rewrite = EquilibriumGraphRewriter([replace_relu], max_use_ratio=10)
    rewrite_graph(x, include=[], custom_rewrite=relu_rewrite)


def test_mlp_change_layer_size():
    """Make sure that we can modify layers functions."""

    x = at.matrix("X")
    x = Dense(100)(x)
    x = relu(x)
    x = Dense(10)(x)

    @node_rewriter([DenseLayer])
    def replace_layer(fgraph, node):
        if isinstance(node.op, DenseLayer):
            x = node.inputs[0]
            W = node.inputs[1]
            b = node.inputs[2]
            units = node.op.units
            if units == 100:
                out = Dense(10, W, b)(x)
                return out.owner.outputs

    layer_rewrite = EquilibriumGraphRewriter([replace_layer], max_use_ratio=10)
    rewrite_graph(x, include=[], custom_rewrite=layer_rewrite)


def test_mlp_add_layer():
    """Make sure that we can add a layer in a model."""

    inp = at.matrix("X")
    x0 = Dense(100)(inp)
    x1 = relu(x0)
    x2 = Dense(10)(x1)

    class AddLayer(GraphRewriter):
        def add_requirements(self, fgraph):
            fgraph.attach_feature(ReplaceValidate())

        def apply(self, fgraph):
            for node in fgraph.toposort():
                if isinstance(node.op, DenseLayer):
                    units = node.op.units
                    if units == 100:
                        out = node.outputs[0]
                        W = at.matrix("new W")
                        b = at.vector("new b")
                        new = Dense(50, W, b)(out)
                        fgraph.replace_validate(out, new, import_missing=True)

    graph = FunctionGraph([inp] + x0.owner.inputs[1:] + x2.owner.inputs[1:], [x2])
    AddLayer().rewrite(graph)
