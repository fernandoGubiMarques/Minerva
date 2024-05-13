from itertools import product
from typing import Any, List, Sequence

import numpy as np
import torch
from perlin_noise import PerlinNoise


class _Transform:
    """This class is a base class for all transforms. Transforms is just a
    fancy word for a function that takes an input and returns an output. The
    input and output can be anything. However, transforms operates over a
    single sample of data and does not require any additional information to
    perform the transformation. The __call__ method should be overridden in
    subclasses to define the transformation logic.
    """

    def __call__(self, *args, **kwargs) -> Any:
        """Implement the transformation logic in this method. Usually, the
        transformation is applyied on a single sample of data.
        """
        raise NotImplementedError


class TransformPipeline(_Transform):
    """Apply a sequence of transforms to a single sample of data and return the
    transformed data.
    """

    def __init__(self, transforms: Sequence[_Transform]):
        """Apply a sequence of transforms to a single sample of data and return
        the transformed data.

        Parameters
        ----------
        transforms : List[_Transform]
            A list of transforms to be applied to the input data.
        """
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        """Apply a sequence of transforms to a single sample of data and return
        the transformed data.
        """
        for transform in self.transforms:
            x = transform(x)
        return x


class Flip(_Transform):
    """Flip the input data along the specified axis."""

    def __init__(self, axis: int | List[int] = 0):
        """Flip the input data along the specified axis.

        Parameters
        ----------
        axis : int | List[int], optional
            One or more axis to flip the input data along, by default 0.
            If a list of axis is provided, the input data is flipped along all the specified axis in the order they are provided.
        """
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Flip the input data along the specified axis.
        if axis is an integer, the input data is flipped along the specified axis.
        if axis is a list of integers, the input data is flipped along all the specified axis in the order they are provided.
        The input must have the same, or less, number of dimensions as the length of the list of axis.
        """

        if isinstance(self.axis, int):
            return np.flip(x, axis=self.axis)

        assert (
            len(self.axis) <= x.ndim
        ), "Axis list has more dimentions than input data. The lenth of axis needs to be less or equal to input dimentions."

        for axis in self.axis:
            x = np.flip(x, axis=axis)

        return x


class PerlinMasker(_Transform):
    """Zeroes entries of a tensor according to the sign of Perlin noise. Seed for the noise generator given by torch.randint"""

    def __init__(self, octaves: int, scale: float = 1):
        """Zeroes entries of a tensor according to the sign of Perlin noise. Seed for the noise generator given by torch.randint

        Parameters
        ----------
        octaves: int
            Level of detail for the Perlin noise generator
        scale: float = 1
            Optionally rescale the Perlin noise. Default is 1 (no rescaling)
        """
        if octaves <= 0:
            raise ValueError(
                f"Number of octaves must be positive, but got {octaves=}"
            )
        if scale == 0:
            raise ValueError(f"Scale can't be 0")
        self.octaves = octaves
        self.scale = scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Zeroes entries of a tensor according to the sign of Perlin noise.

        Parameters
        ----------
        x: np.ndarray
            The tensor whose entries to zero.
        """

        mask = np.empty_like(x, dtype=bool)
        noise = PerlinNoise(self.octaves, torch.randint(0, 2**32, (1,)).item())
        denom = self.scale * max(x.shape)

        for pos in product(*[range(i) for i in mask.shape]):
            mask[pos] = noise([i / denom for i in pos]) < 0

        return x * mask


class Squeeze(_Transform):
    """Remove single-dimensional entries from the shape of an array."""

    def __init__(self, axis: int):
        """Remove single-dimensional entries from the shape of an array.

        Parameters
        ----------
        axis : int
            The position of the axis to be removed.
        """
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Remove single-dimensional entries from the shape of an array."""
        return np.squeeze(x, axis=self.axis)


class Unsqueeze(_Transform):
    """Add a new axis to the input data at the specified position."""

    def __init__(self, axis: int):
        """Add a new axis to the input data at the specified position.

        Parameters
        ----------
        axis : int
            The position of the new axis to be added.
        """
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Add a new axis to the input data at the specified position."""
        return np.expand_dims(x, axis=self.axis)


class CastTo(_Transform):
    """Cast the input data to the specified data type."""

    def __init__(self, dtype: type | str):
        """Cast the input data to the specified data type.

        Parameters
        ----------
        dtype : type
            The data type to which the input data will be cast.
        """
        self.dtype = dtype

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Cast the input data to the specified data type."""
        return x.astype(self.dtype)
