"""Interface for clustering neurons in an MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ClustererInterface:
    """Interface for clustering neurons in an MLP."""

    def cluster(self) -> list[np.ndarray]:
        """Cluster the neurons in the MLP."""
        msg = "Method not implemented. Use a subclass."
        raise NotImplementedError(msg)
