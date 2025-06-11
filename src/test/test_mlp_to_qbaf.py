"""Tests for converting MLP to QBAF."""

from __future__ import annotations

import string

import numpy as np
import pytest

from mlp_to_qbaf_converter.mlp_to_qbaf import MLPToQBAF
from mlp_to_qbaf_converter.utils import forward_pass, logistic


@pytest.mark.parametrize(
    ("num_inputs", "num_outputs", "num_neurons_per_hidden_layer"),
    [
        (1, 1, [2]),
        (2, 5, [2, 4, 6]),
        (5, 10, [5, 10, 20]),
        (10, 15, [25, 1, 6, 10, 20]),
        (20, 20, [30, 10, 15, 20, 15, 10, 5]),
        (30, 25, [50, 60, 70, 80, 90, 100]),
        (40, 30, [60, 80, 60, 100, 20, 50]),
        (50, 35, [100, 200, 50, 100, 300]),
        (60, 40, [100, 500, 600, 50]),
        (70, 45, [100, 200, 300, 400, 500]),
    ],
)
def test_conversion_arguments_named_correctly(
    num_inputs: int,
    num_outputs: int,
    num_neurons_per_hidden_layer: list[int],
) -> None:
    """Test that the arguments of the QBAF are named correctly.

    :param num_inputs: The number of input neurons.
    :param num_outputs: The number of output neurons.
    :param num_neurons_per_hidden_layer: A list of the number of neurons in each\
          hidden layer.
    """
    rng = np.random.default_rng(2025)

    num_hidden_layers = len(num_neurons_per_hidden_layer)
    neurons_per_layer = np.array(
        [num_inputs, *num_neurons_per_hidden_layer, num_outputs],
        dtype=int,
    )

    input_feature_names = [
        "".join(rng.choice(list(string.ascii_lowercase))) for _ in range(num_inputs)
    ]
    output_names = [
        "".join(rng.choice(list(string.ascii_lowercase))) for _ in range(num_outputs)
    ]

    weights = [
        np.zeros((neurons_per_layer[i], neurons_per_layer[i + 1]))
        for i in range(len(neurons_per_layer) - 1)
    ]
    biases = [
        np.zeros(neurons_per_layer[i + 1]) for i in range(len(neurons_per_layer) - 1)
    ]
    input_example = np.ones(num_inputs)

    qbaf = MLPToQBAF(
        neurons_per_layer,
        weights,
        biases,
        "logistic",
        input_feature_names,
        output_names,
        input_example,
    ).get_qbaf()

    # Check that the arguments are named correctly
    hiden_argument_names = [
        f"Layer {layer} Neuron {neuron + 1}"
        for layer in range(1, num_hidden_layers + 1)
        for neuron in range(neurons_per_layer[layer])
    ]
    assert set(qbaf.arguments.keys()) == set(
        input_feature_names + output_names + hiden_argument_names,
    )


@pytest.mark.parametrize(
    ("num_inputs", "num_outputs", "num_neurons_per_hidden_layer", "seed"),
    [
        (1, 1, [2], 1111),
        (2, 5, [2, 4, 6], 2222),
        (5, 10, [5, 10, 20], 3333),
        (10, 15, [25, 1, 6, 10, 20], 4444),
        (20, 20, [30, 10, 15, 20, 15, 10, 5], 5555),
        (30, 25, [50, 60, 70, 80, 90, 100], 6666),
        (40, 30, [60, 80, 60, 100, 20, 50], 7777),
        (50, 35, [100, 200, 50, 100, 300], 8888),
        (60, 40, [100, 500, 600, 50], 9999),
        (70, 45, [100, 200, 300, 400, 500], 11111),
    ],
)
def test_conversion_strengths(
    num_inputs: int,
    num_outputs: int,
    num_neurons_per_hidden_layer: list[int],
    seed: int,
) -> None:
    """Test that the strengths are equal to the activations of the MLP.

    :param num_inputs: The number of input neurons.
    :param num_outputs: The number of output neurons.
    :param num_neurons_per_hidden_layer: A list of the number of neurons in each\
          hidden layer.
    :param seed: The seed for the random number generator.
    """
    rng = np.random.default_rng(seed)

    num_hidden_layers = len(num_neurons_per_hidden_layer)
    neurons_per_layer = np.array(
        [num_inputs, *num_neurons_per_hidden_layer, num_outputs],
        dtype=int,
    )

    # Multiply by random integer so weights and biases not all in range [0, 1)
    weights = [
        rng.random((neurons_per_layer[i], neurons_per_layer[i + 1]))
        * rng.integers(1, 10)
        for i in range(len(neurons_per_layer) - 1)
    ]
    biases = [
        rng.random(neurons_per_layer[i + 1]) * rng.integers(1, 10)
        for i in range(len(neurons_per_layer) - 1)
    ]
    input_example = rng.random(num_inputs)

    input_feature_names = [f"input_{i}" for i in range(num_inputs)]
    output_names = [f"output_{i}" for i in range(num_outputs)]

    qbaf = MLPToQBAF(
        neurons_per_layer,
        weights,
        biases,
        "logistic",
        input_feature_names,
        output_names,
        input_example,
    ).get_qbaf()

    # Perform forward pass with the MLP
    activations = forward_pass(input_example, weights, biases, logistic)

    # Create a similar array, mapping the QBAF strengths to the activations
    qbaf_strengths = [np.zeros_like(layer) for layer in activations]

    for i in range(num_inputs):
        qbaf_strengths[0][i] = qbaf.arguments[f"input_{i}"].strength

    for layer in range(1, num_hidden_layers + 1):
        for neuron in range(neurons_per_layer[layer]):
            qbaf_strengths[layer][neuron] = qbaf.arguments[
                f"Layer {layer} Neuron {neuron + 1}"  # How hidden neurons are named
            ].strength

    for i in range(num_outputs):
        qbaf_strengths[-1][i] = qbaf.arguments[f"output_{i}"].strength

    for layer in range(len(activations)):
        assert np.allclose(activations[layer], qbaf_strengths[layer])
