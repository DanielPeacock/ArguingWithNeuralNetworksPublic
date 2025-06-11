"""Contains the LocalSpArX class, which is used to sparsify an MLP."""

from __future__ import annotations

from types import MappingProxyType

import numpy as np

import Uncertainpy.src.uncertainpy.gradual as grad
from mlp_to_qbaf_converter.argument_attribution_explanation import AAE
from mlp_to_qbaf_converter.errors import ActivationFunctionNotImplementedError
from mlp_to_qbaf_converter.mlp_to_qbaf import MLPToQBAF
from mlp_to_qbaf_converter.utils import (
    forward_pass,
    logistic,
    relu,
)
from sparx.aaes_clusterer import AAEClusterer
from sparx.activations_clusterer import ActivationsClusterer
from sparx.local_merger import LocalMerger


class TaskType:
    """Enum for the task type of the MLP."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ClusteringMethod:
    """Enum for the clustering method to use."""

    AAE_GRADIENT = "aae_gradient"
    AAE_REMOVAL = "aae_removal"
    AAE_SHAP = "aae_shap"
    ACTIVATIONS = "activations"


class LocalSpArX:
    """Class to sparsify an MLP."""

    SUPPORTED_ACTIVATIONS = MappingProxyType(
        {"logistic": logistic, "relu": relu},
    )

    def __init__(  # noqa: PLR0913
        self,
        weights: list[list[np.ndarray]],
        biases: list[np.ndarray],
        activation_func_name: str,
        shrinkage_percentage: int,
        example_row: np.ndarray,
        train_set: np.ndarray,
        kernel_width: float,
        input_feature_names: list[str] | None = None,
        output_arg_names: list[str] | None = None,
        topic_arg_name: str | None = None,
        cluster_method: ClusteringMethod = ClusteringMethod.ACTIVATIONS,
        task_type: TaskType = TaskType.CLASSIFICATION,
    ) -> None:
        """Initialise the SpArX class.

        :param weights: A list containing a list for each layer each with an \
        array of weights for the neurons in that layer \
        (not including the output layer).
        :param biases: A list containing an array of biases for each layer \
        (not including the input layer).\
        :param activation_func_name: The name of the activation function to use.\
        :param shrinkage_percentage: The percentage to reduce the size of the MLP by.
        :param example_row: The example row to use for generating the local dataset and\
        produce explanations for (include expected output label).
        :param train_set: The training set to use for generating the local dataset.
        :param kernel_width: The width of the kernel function.
        :param input_feature_names: The names of the input features. Required if\
            cluster_method is AAE. Ignored otherwise.
        :param output_arg_names: The names of the output arguments. Required if\
            cluster_method is AAE. Ignored otherwise.
        :param topic_arg_name: The name of the topic argument. Required if\
            cluster_method is AAE. Ignored otherwise.
        :param cluster_method: The method to use for clustering the neurons in the MLP.
        :task_type: The type of task the MLP is used for (classification or regression).
        """
        if activation_func_name not in self.SUPPORTED_ACTIVATIONS:
            msg = f"Activation function {activation_func_name} is not supported.\
            Supported activations are: {self.SUPPORTED_ACTIVATIONS}"
            raise ActivationFunctionNotImplementedError(
                msg,
                list(self.SUPPORTED_ACTIVATIONS.keys()),
            )

        activation_func = self.SUPPORTED_ACTIVATIONS[activation_func_name]

        if shrinkage_percentage <= 0 or shrinkage_percentage >= 100:  # noqa: PLR2004
            msg = "Shrinkage percentage must be between 0 and 100 (exclusive)."
            raise ValueError(msg)

        self.shrinkage_percentage = shrinkage_percentage
        self.preserve_percentage = 100 - shrinkage_percentage
        self.example = np.asarray(example_row[:-1])
        self.example_row = np.asarray(example_row)

        if kernel_width <= 0:
            msg = "Kernel width must be greater than 0."
            raise ValueError(msg)

        self.kernel_width = kernel_width

        activations = forward_pass(
            self.example,
            weights,
            biases,
            activation_func,
        )

        if cluster_method != ClusteringMethod.ACTIVATIONS:
            if (
                input_feature_names is None
                or output_arg_names is None
                or topic_arg_name is None
            ):
                msg = "Input feature names, output argument names and topic argument\
                    name must be provided when using AAE clustering."
                raise ValueError(msg)

            neurons_per_layer = [len(weights[0])] + [len(b) for b in biases]
            qbaf = MLPToQBAF(
                neurons_per_layer,
                weights,
                biases,
                activation_func_name,
                input_feature_names,
                output_arg_names,
                self.example,
            ).get_qbaf()

        match cluster_method:
            case ClusteringMethod.ACTIVATIONS:
                clusterer = ActivationsClusterer(activations, self.preserve_percentage)

            case ClusteringMethod.AAE_GRADIENT:
                aae = AAE(
                    qbaf,
                    grad.SumAggregation(),
                    grad.MLPBasedInfluence(),
                    topic_arg_name,
                    1e-8,
                    verbose=False,
                    do_shap=False,
                    do_removal=False,
                    do_gradient=True,
                )
                aae_scores = aae.get_gradients()
                aae_scores[topic_arg_name] = 0
                clusterer = AAEClusterer(
                    aae_scores,
                    self.preserve_percentage,
                    output_arg_names,
                    biases,
                )

            case ClusteringMethod.AAE_REMOVAL:
                aae = AAE(
                    qbaf,
                    grad.SumAggregation(),
                    grad.MLPBasedInfluence(),
                    topic_arg_name,
                    1e-8,
                    verbose=False,
                    do_shap=False,
                    do_removal=True,
                    do_gradient=False,
                )
                aae_scores = aae.get_removal_scores()
                aae_scores[topic_arg_name] = 0
                clusterer = AAEClusterer(
                    aae_scores,
                    self.preserve_percentage,
                    output_arg_names,
                    biases,
                )
            case ClusteringMethod.AAE_SHAP:
                aae = AAE(
                    qbaf,
                    grad.SumAggregation(),
                    grad.MLPBasedInfluence(),
                    topic_arg_name,
                    1e-8,
                    verbose=False,
                    do_shap=True,
                    do_removal=False,
                    do_gradient=False,
                )
                aae_scores = aae.get_shap_scores()
                aae_scores[topic_arg_name] = 0
                clusterer = AAEClusterer(
                    aae_scores,
                    self.preserve_percentage,
                    output_arg_names,
                    biases,
                )

            case _:
                msg = f"Clustering method {cluster_method} is not supported."
                raise ValueError(msg)

        self.cluster_labels = clusterer.cluster()

        merger = LocalMerger(
            weights,
            biases,
            activation_func,
            train_set,
            example_row,
            kernel_width,
            task_type,
            self.cluster_labels,
        )

        self.example_weights = merger.example_weights
        self.local_dataset = merger.local_dataset
        self.sparsified_weights, self.sparsified_biases = merger.merge()
        self.sparsified_shape = [len(example_row) - 1] + [
            len(b) for b in self.sparsified_biases
        ]

    def get_sparsified_mlp(self) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        """Get the sparsified MLP.

        Return:
        The sparsified MLP. A tuple of weights and biases.

        """
        return self.sparsified_weights, self.sparsified_biases

    def get_sparsified_shape(self) -> list[int]:
        """Get the shape of the sparsified MLP.

        Return:
        The shape of the sparsified MLP.

        """
        return self.sparsified_shape

    def get_containing_neurons(
        self,
        layer_num: int,
        neuron_num: int,
    ) -> np.ndarray[int]:
        """Get the neurons in each cluster (hidden layers).

        :param: layer_num: The layer number (indexed from 1).
        :param: The neuron number (indexed from 1).

        Return:
        The neuron indexes in each cluster.

        """
        return (
            np.argwhere(
                self.cluster_labels[layer_num - 1] == neuron_num - 1,
            ).flatten()
            + 1
        )
