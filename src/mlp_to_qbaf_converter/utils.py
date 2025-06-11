"""Activation functions used in the MLP to QBAF converter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import networkx as nx
import numpy as np
from graphviz import Digraph

if TYPE_CHECKING:
    import Uncertainpy.src.uncertainpy.gradual as grad


def logistic(x: np.ndarray | float) -> np.ndarray | float:
    """Apply the logistic function."""
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray | float) -> np.ndarray | float:
    """Apply the ReLU function."""
    return np.maximum(0, x)


def softmax(x: np.ndarray) -> np.ndarray:
    """Apply the softmax function."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def forward_pass(
    example: np.ndarray,
    weights: list[np.ndarray[np.ndarray]],
    biases: list[np.ndarray],
    activation_func: Callable,
    apply_softmax: bool = False,  # noqa: FBT001, FBT002
) -> list[np.ndarray]:
    """Perform a forward pass of the MLP (logistic activation).

    :param example: The example to perform the forward pass on.
    :param weights: The weights of the MLP.
    :param biases: The biases of the MLP.
    :param activation_func: The activation function to use.
    :param apply_softmax: Whether to apply the softmax function to the output.

    Return:
    The activations at each neuron.

    """
    activations = []
    activations.append(example)

    for layer in range(len(weights)):
        z = activations[-1] @ weights[layer] + biases[layer]
        activations.append(activation_func(z))

    if apply_softmax:
        activations[-1] = softmax(activations[-1])
    return activations


def forward_pass_dataset(
    dataset: np.ndarray,
    weights: list[np.ndarray[np.ndarray]],
    biases: list[np.ndarray],
    activation_func: Callable,
    apply_softmax: bool = False,  # noqa: FBT001, FBT002
) -> list[np.ndarray]:
    """Perform a forward pass of the MLP (logistic activation) on a dataset.

    :param dataset: The dataset to perform the forward pass on.
    :param weights: The weights of the MLP.
    :param biases: The biases of the MLP.
    :param activation_func: The activation function to use.
    :param apply_softmax: Whether to apply the softmax function to the output.

    Return:
    The activations at each neuron.

    """
    activations = []
    activations.append(dataset)

    for layer in range(len(weights)):
        layer_activations = []

        for prev in activations[-1]:
            z = prev @ weights[layer] + biases[layer]
            layer_activations.append(activation_func(z))

        layer_activations = np.array(layer_activations)
        activations.append(layer_activations)

    if apply_softmax:
        for i in range(len(activations[-1])):
            activations[-1][i] = softmax(activations[-1][i])

    return activations


def kernel(x: np.ndarray, kernel_width: float) -> np.ndarray:
    """Apply the kernel function."""
    return np.sqrt(np.exp(-(x**2) / kernel_width**2))


def plot_qbaf(qbaf: grad.BAG, edge_weights: bool = True) -> Digraph:  # noqa: FBT001, FBT002
    """Create a directed graph using Graphviz.

    :param qbaf: The QBAF to plot.
    :param edge_weights: Whether to include edge weights in the graph.

    Returns:
    The directed graph object.

    """
    qbaf_graph = Digraph()
    qbaf_graph.attr(rankdir="LR")
    qbaf_graph.attr(overlap="false")  # avoid node overlaps
    qbaf_graph.attr(splines="true")  # use curved edges
    qbaf_graph.attr(margin="0")  # reduce page margin
    qbaf_graph.attr(pad="0")  # padding around graph
    qbaf_graph.attr(nodesep="0.2")
    qbaf_graph.attr(ranksep="0.2")

    qbaf_graph.attr(
        "node",
        fontsize="12",
        width="0",
        height="0",
        margin="0.01,0.01",  # very tight node margins
        style="filled",
        fillcolor="lightgrey",
    )

    # Sort arguments by layer and neuron

    for arg in qbaf.arguments.values():
        qbaf_graph.node(
            arg.name,
            label=f"{arg.name}\nStrength:{arg.strength:.4f}",
        )

    for attack in qbaf.attacks:
        attack_weight = round(attack.attacked.attackers[attack.attacker], 2)
        qbaf_graph.edge(
            attack.attacker.name,
            attack.attacked.name,
            style="solid",
            color="red",
            label=f"{attack_weight}" if edge_weights else "",
        )

    for support in qbaf.supports:
        support_weight = round(support.supported.supporters[support.supporter], 2)
        qbaf_graph.edge(
            support.supporter.name,
            support.supported.name,
            style="dashed",
            color="blue",
            label=f"{support_weight}" if edge_weights else "",
        )

    return qbaf_graph


def generate_nx_graph(qbaf: grad.BAG) -> nx.DiGraph:
    """Generate a NetworkX graph from a QBAF.

    :param qbaf: The QBAF to convert.

    Returns:
    The NetworkX graph object.

    """
    graph = nx.DiGraph()

    for arg in qbaf.arguments.values():
        graph.add_node(arg.name, fill_color="#00b4d9", name=arg.name)

    for attack in qbaf.attacks:
        graph.add_edge(
            attack.attacker.name,
            attack.attacked.name,
            edge_color="red",
            style="solid",
            edge_weight=attack.attacked.attackers[attack.attacker],
        )
    for support in qbaf.supports:
        graph.add_edge(
            support.supporter.name,
            support.supported.name,
            edge_color="blue",
            style="dashed",
            edge_weight=support.supported.supporters[support.supporter],
        )

    return graph
