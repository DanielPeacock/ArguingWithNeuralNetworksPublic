"""Errors for the MLPToQBAF package."""

from __future__ import annotations


class MLPConversionError(Exception):
    """Generic class for errors when converting an MLP to QBAF."""

    def __init__(self, msg: str) -> None:
        """Construct error with MLP to QBAF conversions.

        :param msg: message to represent the error.
        """
        super().__init__(msg)


class DimensionError(MLPConversionError):
    """Class for errors if arguments with unmatched dimensions are given."""

    def __init__(self, msg: str | None = None) -> None:
        """Construct dimension error with MLP to QBAF conversions.

        :param msg: Message to display. Displays 'Dimension error' if None.
        """
        if not msg:
            msg = "Dimension error."

        super().__init__(msg)


class ActivationFunctionNotImplementedError(MLPConversionError):
    """Class for errors if arguments with unmatched dimensions are given."""

    def __init__(self, function: str, supported: list[str]) -> None:
        """Construct dimension error with MLP to QBAF conversions.

        :param msg: message to represent the error.
        """
        super().__init__(
            f"Activation function {function} not supported. \
Supported functions are: {supported}",
        )


class ExplanationError(Exception):
    """Generic class for errors when producing explanations."""

    def __init__(self, msg: str) -> None:
        """Construct error with MLP to QBAF conversions.

        :param msg: message to represent the error.
        """
        super().__init__(msg)


class CounterfactualExplanationNotFoundError(ExplanationError):
    """Class for errors if explanations are not found."""

    def __init__(self, msg: str | None) -> None:
        """Construct error with MLP to QBAF conversions.

        :param msg: message to represent the error.
        """
        if msg is None:
            msg = (
                "Counterfactual explanation not found. May need to increase M or delta."
            )

        super().__init__(msg)
