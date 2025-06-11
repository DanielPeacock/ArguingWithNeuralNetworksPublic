"""Merger interface for merging neurons in an MLP."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class MergerInterface:
    """Interface for merging neurons in an MLP."""

    def merge(self) -> tuple[list[list[np.ndarray]], list[np.ndarray]]:
        """Merge the neurons in the MLP."""
        msg = "Method not implemented. Use a subclass."
        raise NotImplementedError(msg)
