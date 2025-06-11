"""ActivationsClusterer class."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class ActivationsClusterer:
    """Cluster the neurons in the MLP based on their activations."""

    def __init__(
        self,
        activations: list[np.ndarray],
        preserve_percentage: int,
        seed: int = 2025,
    ) -> None:
        """Initialise the ActivationsClusterer class.

        :param activations: The activations of the neurons in the MLP.
        :param preserve_percentage: The percentage of neurons to preserve.
        :param seed: The random seed for the clustering algorithm.
        """
        super().__init__()
        self.activations = activations
        self.preserve_percentage = preserve_percentage
        self.seed = seed

    def cluster(self) -> list[np.ndarray]:
        """Cluster the neurons in the MLP.

        :param layer: The layer to cluster (activations of neurons).

        Return:
        A list of arrays of the clusters each neuron belongs to.

        """
        clusters = []
        for layer in self.activations[1:]:
            labels = (
                KMeans(
                    n_clusters=max(len(layer) * self.preserve_percentage // 100, 1),
                    random_state=self.seed,
                )
                .fit(np.array(layer).reshape(-1, 1))
                .labels_
            )
            labels = np.unique(labels, return_inverse=True)[1]
            clusters.append(labels)

        return clusters
