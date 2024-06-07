from typing import Any, Callable, Optional, Sequence
from numpy.typing import ArrayLike
import numpy as np

from minerva.data.readers import _Reader


class MultiReader(_Reader):
    """Reader that composes items from other readers.
    
    Its i-th item is the i-th item of each of the child-readers merged
    together according to a collate_fn function."""

    def __init__(
        self,
        readers: Sequence[_Reader],
        collate_fn: Optional[Callable] = None
    ):
        """Collects data from multiple readers and collates them
        
        Parameters
        ----------
        readers: Sequence[_Reader]
            The readers from which the data will be collected. At least one must be
            provided. If the readers have different lengths, data will only be
            collected up until the length of the smallest child-reader.
        collate_fn: Callable
            A function that recieves a list of items read from the child-readers and
            returns a single item for this reader. Defaults to numpy.stack, which
            means it is not optional if the child-readers are not returning single
            numpy arrays.
        """
        assert len(readers) > 0, "MultiReader expects at least one reader as argument."
        
        self._length = min(len(reader) for reader in readers)
        self._readers = readers
        self.collate_fn = collate_fn or np.stack
    
    def __len__(self) -> int:
        """Returns the length of the smallest child-reader
        
        Returns
        -------
        int
            The length of the reader."""
        return self._length
    
    def __getitem__(self, index: int) -> Any:
        """Retrieves the items from each reader at the specified index and collates them accordingly.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        Any
            An item from the reader.
        """
        return self.collate_fn([reader[index] for reader in self._readers])


