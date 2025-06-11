"""LocalMerger class to merge the weights and biases in an MLP locally."""

from __future__ import annotations

from functools import partial
from typing import Callable

import numpy as np
import sklearn

from mlp_to_qbaf_converter.utils import forward_pass_dataset, kernel
from sparx.lime.lime.lime_tabular import LimeTabularExplainer
from sparx.merger import MergerInterface


class LocalMerger(MergerInterface):
    """Merge the weights and biases in an MLP locally."""

    def __init__(  # noqa: PLR0913
        self,
        weights: list[list[np.ndarray]],
        biases: list[np.ndarray],
        activation_func: Callable,
        train_set: np.ndarray,
        example_row: np.ndarray,
        kernel_width: float,
        task_type: str,
        cluster_labels: list[np.ndarray],
    ) -> None:
        """Initialise the LocalMerger.

        :param weights: A list containing a list for each layer each with an \
        array of weights for the neurons in that layer \
        (not including the output layer).
        :param biases: A list containing an array of biases for each layer \
        (not including the input layer).\
        :param activation_func: The activation function to use.
        :param train_set: The training set to use for generating the local dataset.
        :param example_row: The example row to use for generating the local dataset and\
        produce explanations for (include expected output label).
        :param kernel_width: The width of the kernel function.
        :task_type: The type of task the MLP is used for (classification or regression).
        :param cluster_labels: The cluster labels for each layer.
        """
        super().__init__()

        self.weights = weights
        self.num_hidden_layers = len(self.weights) - 1
        self.biases = biases
        self.activation_func = activation_func
        self.train_set = train_set
        self.example_row = example_row
        self.kernel_width = kernel_width
        self.kernel_func = partial(kernel, kernel_width=kernel_width)
        self.task_type = task_type
        self.cluster_labels = cluster_labels

        self.local_dataset, self.example_weights = self.generate_local_dataset()
        self.activations_dataset = forward_pass_dataset(
            self.local_dataset[:, :-1],
            self.weights,
            self.biases,
            self.activation_func,
        )

    def merge(self) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        """Merge the biases and weights in the MLP.

        Returns:
        The sparsified weights and biases.

        """
        merged_weights = []
        merged_biases = []

        partial_weights = [self.weights[0]]

        for i in range(self.num_hidden_layers):
            n_clusters = max(self.cluster_labels[i]) + 1
            merged_weights.append(
                self._merge_weights(
                    i,
                    n_clusters,
                    partial_weights[i],
                ),
            )
            merged_biases.append(self._merge_biases(i, n_clusters))

            partial_weights.append(
                self._update_partial_weights(
                    i,
                    merged_weights,
                    merged_biases,
                    n_clusters,
                ),
            )

        merged_weights.append(partial_weights[-1])
        merged_biases.append(self.biases[-1])

        return merged_weights, merged_biases

    def _merge_weights(
        self,
        layer: int,
        n_clusters: int,
        partial_weights: np.ndarray,
    ) -> list[np.ndarray]:
        """Merge the weights in a layer.

        :param layer: The layer to merge.
        :param n_clusters: The number of clusters in the layer.
        :param partial_weights: The weights in the layer.

        Return:
        The merged weights.

        """
        merged_weights = [
            np.mean(partial_weights.T[self.cluster_labels[layer] == label], axis=0)
            for label in range(n_clusters)
        ]

        return np.asarray(merged_weights).T

    def _merge_biases(
        self,
        layer: int,
        n_clusters: int,
    ) -> np.ndarray:
        """Merge the biases in a layer.

        :param layer: The layer to merge.
        :param n_clusters: The number of clusters in the layer.

        Return:
        The merged biases.

        """
        merged_biases = [
            np.mean(self.biases[layer][self.cluster_labels[layer] == label])
            for label in range(n_clusters)
        ]

        return np.asarray(merged_biases)

    def _update_partial_weights(
        self,
        layer: int,
        merged_weights: list[list[np.ndarray]],
        merged_biases: list[np.ndarray],
        n_clusters: int,
    ) -> np.ndarray:
        """Update the partial weights.

        :param layer: The layer to update.
        :param merged_weights: The merged weights.
        :param merged_biases: The merged biases.
        :param n_clusters: The number of clusters in the layer.

        Return:
        The updated partial weights.

        """
        new_partial_weights = []
        partial_activations = forward_pass_dataset(
            self.local_dataset[:, :-1],
            merged_weights,
            merged_biases,
            self.activation_func,
        )[layer + 1]

        new_example_weights = self.example_weights / np.sum(self.example_weights)

        for label in range(n_clusters):
            h_star = partial_activations[:, label]
            all_hidden_activations = self.activations_dataset[layer + 1][
                :,
                self.cluster_labels[layer] == label,
            ]

            h_star = np.array([1 if hs == 0 else hs for hs in list(h_star)])

            normalised_activations = np.sum(
                np.multiply(all_hidden_activations.T, new_example_weights / h_star).T,
                axis=0,
            )

            new_partial_weights.append(
                np.dot(
                    normalised_activations,
                    self.weights[layer + 1][self.cluster_labels[layer] == label],
                ),
            )

        return np.asarray(new_partial_weights)

    def generate_local_dataset(self) -> np.ndarray:
        """Generate a local dataset.

        Return:
        The local dataset.

        """
        explainer = LimeTabularExplainer(
            self.train_set,
            mode=self.task_type,
            random_state=123,
        )
        n_samples = 5000
        data, inverse = explainer.data_inverse(self.example_row, n_samples, "gaussian")
        scaled_data = (data - explainer.scaler.mean_) / explainer.scaler.scale_
        output_neurons = forward_pass_dataset(
            inverse[:, :-1],
            self.weights,
            self.biases,
            self.activation_func,
            apply_softmax=True,
        )[-1]
        data_labels = np.argmax(output_neurons, axis=1)

        distance_to_target = sklearn.metrics.pairwise_distances(
            scaled_data,
            scaled_data[0].reshape(1, -1),
            metric="euclidean",
        ).ravel()

        example_weights = self.kernel_func(distance_to_target)

        used_features = explainer.base.feature_selection(
            scaled_data,
            data_labels,
            example_weights,
            self.train_set.shape[1],
            "auto",
        )

        return scaled_data[:, used_features], example_weights
