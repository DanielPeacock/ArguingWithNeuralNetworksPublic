"""Convert an MLP to a QBAF."""

from __future__ import annotations

import numpy as np

import Uncertainpy.src.uncertainpy.gradual as grad
from mlp_to_qbaf_converter.errors import (
    ActivationFunctionNotImplementedError,
    DimensionError,
)
from mlp_to_qbaf_converter.utils import logistic


class MLPToQBAF:
    """Class to convert MLP to QBAF."""

    def __init__(  # noqa: PLR0913
        self,
        neurons_per_layer: np.ndarray,
        weights: list[list[np.ndarray]],
        biases: list[np.ndarray],
        activation_func_name: str,
        input_feature_names: list[str],
        output_names: list[str],
        input_example: list[float],
    ) -> None:
        """Initialise conversion from MLP to QBAF.

        :param neurons_per_layer: an array with the number of neurons in each layer \
            (including input and output layers).
        :param weights: A list containing a list for each layer each with an \
            array of weights for the neurons in that layer \
                (not including the output layer).
        :param biases: A list containing an array of biases for each layer \
            (not including the input layer).\
        :param activation_func_name: A string with the name of the activation\
            function used to train the MLP.
        :param input_feature_names: List of input feature names.
        :param output_names: List of output names.
        :param input_example: The input example to create a QBAF for.

        Raises:
        MLPConversionError if there is an error with dimensions, or an invalid
        activation function is given.

        """
        self.supported_funcs = {"logistic": logistic}

        self.no_layers = len(neurons_per_layer)
        self.neurons_per_layer = neurons_per_layer
        self.weights = weights
        self.biases = biases
        self.activation_func_name = activation_func_name
        self.input_feature_names = input_feature_names
        self.output_names = output_names
        self.input_example = input_example

        self.error_checks()

        self.activation_func = self.supported_funcs[self.activation_func_name]
        self.bag = self.create_qbaf()

    def error_checks(self) -> None:
        """Check for errors in class attributes.

        Raises:
            MLPConversionError if there is an error.

        """
        self.check_weight_dimensions()
        self.check_bias_dimensions()

        if self.activation_func_name not in self.supported_funcs:
            raise ActivationFunctionNotImplementedError(
                self.activation_func_name,
                self.supported_funcs,
            )

        if len(self.input_example) != self.neurons_per_layer[0]:
            msg = "Number of input values does \
                not equal the number of input neurons."
            raise DimensionError(msg)

        if len(self.input_feature_names) != self.neurons_per_layer[0]:
            msg = "Number of input feature names does not\
                match the number of input neurons."
            raise DimensionError(msg)

        if len(self.output_names) != self.neurons_per_layer[-1]:
            msg = "Number of output names does not match the number of output neurons."
            raise DimensionError(msg)

    def check_weight_dimensions(self) -> None:
        """Check the dimensions of the weights given.

        Raises:
            MLPConversionError if there is an error.

        """
        # Minus 1 since we have weights for the input layer,
        # but not for the output layer.
        if self.no_layers - 1 != len(self.weights):
            msg = "Number of layers (including input layer, and not output) \
does not match the number of weights given."
            raise DimensionError(msg)

        for i in range(len(self.weights)):
            # Check that the number of weight arrays given for each layer
            # matches the number of neurons in the layer.
            if len(self.weights[i]) != self.neurons_per_layer[i]:
                msg = f"Number of weight arrays in layer {i} \
does not match the number of neurons in the layer."
                raise DimensionError(msg)

            # Check the number of weights for each neuron in layer
            # matches the number of neurons in the next layer.
            neurons_next_layer = self.neurons_per_layer[i + 1]
            for j in range(len(self.weights[i])):
                if len(self.weights[i][j]) != neurons_next_layer:
                    msg = f"Number of weights in layer {i}, neuron {j} \
does not match the number of neurons in the next layer."
                    raise DimensionError(msg)

    def check_bias_dimensions(self) -> None:
        """Check the dimensions of the biases given.

        Raises:
            MLPConversionError if there is an error.

        """
        # Minus 1 since we have biases for the output layer,
        # but not for the input layer.
        if self.no_layers - 1 != len(self.biases):
            msg = "Number of bias arrays does not match the number of layers \
(not including the input layer.)"
            raise DimensionError(msg)

        for i in range(len(self.biases)):
            # Check neurons in next layer since the input layer does not have
            # biases.
            if len(self.biases[i]) != self.neurons_per_layer[i + 1]:
                msg = f"Number of neurons in layer {i + 1} \
does not match the number of biases given {len(self.biases[i])}"
                raise DimensionError(msg)

    def create_qbaf(self) -> grad.BAG:
        """Create a QBAF from the class attributes given.

        Returns:
            The QBAF.

        """
        bag = grad.BAG()

        arguments = self.create_arguments()

        self.add_relations(bag, arguments)

        grad.algorithms.computeStrengthValues(
            bag,
            grad.SumAggregation(),
            grad.MLPBasedInfluence(),
        )

        return bag

    def create_arguments(self) -> list[np.ndarray[grad.Argument]]:
        """Create the arguments required for the QBAF.

        Returns:
        A list containing an array of Arguments for each layer.

        """
        arguments = []
        input_arguments = np.array([None] * self.neurons_per_layer[0])

        for i, input_score in enumerate(self.input_example):
            input_arguments[i] = grad.Argument(
                self.input_feature_names[i],
                float(input_score),
            )

        arguments.append(input_arguments)

        base_scores = [self.activation_func(bias) for bias in self.biases]

        for i, scores in enumerate(base_scores[:-1]):
            layer_arguments = np.array([None] * len(scores))
            for j, s in enumerate(scores):
                layer_arguments[j] = grad.Argument(
                    f"Layer {i + 1} Neuron {j + 1}",
                    float(s),
                )

            arguments.append(layer_arguments)

        output_arguments = np.array([None] * len(self.output_names))
        for i, score in enumerate(base_scores[-1]):
            output_arguments[i] = grad.Argument(self.output_names[i], float(score))

        arguments.append(output_arguments)

        return arguments

    def add_relations(
        self,
        bag: grad.BAG,
        arguments: list[np.ndarray[grad.Argument]],
    ) -> None:
        """Add attacks and supports to the BAG using weights."""
        for i in range(len(arguments) - 1):
            for j, argument in enumerate(arguments[i]):
                for k, next_argument in enumerate(arguments[i + 1]):
                    weight = self.weights[i][j][k]
                    if weight < 0:
                        bag.add_attack(argument, next_argument, abs(weight))
                    else:
                        bag.add_support(argument, next_argument, weight)

    def get_qbaf(self) -> grad.BAG:
        """Return the QBAF."""
        return self.bag
