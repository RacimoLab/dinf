import logging
import pathlib

import zarr

logger = logging.getLogger(__name__)


class Cache:
    """
    Wrapper around zarr for managing cached data.
    """

    def __init__(self, path, keys: tuple[str]):
        self.path = pathlib.Path(path)
        self.keys = keys

    def exists(self) -> bool:
        """Returns True if the cache exists, False otherwise."""
        return self.path.exists()

    def load(self) -> tuple:
        """Load data from the cache."""
        if not self.exists():
            raise RuntimeError(f"{self.path} doesn't exist")
        logger.debug(f"Loading data from {self.path}.")
        store = zarr.load(str(self.path))
        return tuple(store[k] for k in self.keys)

    def save(self, data) -> None:
        """Save data to the cache."""
        assert isinstance(data, (tuple))
        if len(data) != len(self.keys):
            raise ValueError(
                f"len(data) != len(self.keys) ({len(data)} != {len(self.keys)})"
            )
        logger.debug(f"Caching data to {self.path}.")
        data_kwargs = dict()
        max_chunk_size = 2 ** 30  # 1 Gb
        for k, v in zip(self.keys, data):
            shape = list(v.shape)
            size = v.size * v.itemsize
            if size > max_chunk_size:
                shape[0] = int(shape[0] * max_chunk_size / size)
            data_kwargs[k] = zarr.array(v, chunks=shape)
        zarr.save(str(self.path), **data_kwargs)
