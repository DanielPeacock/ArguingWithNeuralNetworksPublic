"""AAEClusterer class for clustering using AAE scores."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from sparx.cluster import ClustererInterface


class AAEClusterer(ClustererInterface):
    """Cluster the neurons in the MLP based on their AAE scores."""

    def __init__(
        self,
        aae_scores: dict[str, float],
        preserve_percentage: int,
        output_arg_names: list[str],
        biases: list[np.ndarray],
    ) -> None:
        """Initialise the AAEClusterer class.

        :param aae_scores: The AAE scores of the neurons in the MLP.
        :param preserve_percentage: The percentage of neurons to preserve.
        """
        super().__init__()
        self.aae_scores = aae_scores
        self.preserve_percentage = preserve_percentage
        self.biases = biases
        self.output_arg_names = output_arg_names

    def cluster(self) -> list[np.ndarray]:
        """Cluster the neurons in the MLP based on their AAE scores.

        Return:
        A list of arrays of the clusters each neuron belongs to.

        """
        aae_scores_arr = self.aae_scores_to_array()

        clusters = []

        for layer in aae_scores_arr:
            labels = (
                KMeans(n_clusters=max(len(layer) * self.preserve_percentage // 100, 1))
                .fit(np.array(layer).reshape(-1, 1))
                .labels_
            )
            labels = np.unique(labels, return_inverse=True)[1]
            clusters.append(labels)

        return clusters

    def aae_scores_to_array(self) -> list[np.ndarray]:
        """Convert the AAE scores to an array."""
        last_layer = len(self.biases)
        for i, name in enumerate(self.output_arg_names):
            self.aae_scores[f"Layer {last_layer} Neuron {i + 1}"] = self.aae_scores.pop(
                name,
            )

        aae_scores_list = []

        for i, layer in enumerate(self.biases):
            layer_scores = np.zeros(len(layer))
            for j, _ in enumerate(layer):
                layer_scores[j] = self.aae_scores[f"Layer {i + 1} Neuron {j + 1}"]
            aae_scores_list.append(layer_scores)

        return aae_scores_list
