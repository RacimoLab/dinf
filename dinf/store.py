from __future__ import annotations
import collections.abc
import pathlib


class Store(collections.abc.Sequence):
    """
    A sequence of folders with names "0", "1", ...

    This class implements the :class:`collections.abc.Sequence` protocol.

    :ivar pathlib.Path base:
        Base directory containing the sequence.
    """

    def __init__(self, base: str | pathlib.Path, *, create: bool = False):
        """
        :param base:
            The base directory containing the sequence.
        :param create:
            If True, create the base directory if it doesn't exist.
            If False (*default*), raise an error if the base directory
            doesn't exist.
        """
        self._length = 0
        self.base = pathlib.Path(base)
        if create:
            self.base.mkdir(parents=True, exist_ok=True)
        if not self.base.exists():
            raise ValueError(f"{self.base} not found")
        if not self.base.is_dir():
            raise ValueError(f"{self.base} is not a directory")

        # Find the length.
        try:
            while True:
                path = self[self._length]
                if not path.is_dir():
                    raise ValueError(f"{path} is not a directory")
                self._length += 1
        except IndexError:
            pass

    def __len__(self):
        """
        Length of the sequence.
        """
        return self._length

    def __getitem__(self, index):
        """
        Get the path with the given index.

        :param int index:
            The 0-based index into the sequence.
        """
        if not isinstance(index, int):
            # Slicing not supported.
            raise TypeError("Store index must be an integer")
        if index < 0:
            index += len(self)
        path = self.base / f"{index}"
        if not path.exists():
            raise IndexError(f"{path} not found")
        return path

    def __str__(self):
        s = str(self.base)
        length = len(self)
        if length == 1:
            s += "/0"
        elif length > 0:
            s += f"/{{0..{length - 1}}}"
        return s

    def increment(self):
        """
        Add a new folder to the end of the sequence.
        """
        new = self.base / f"{len(self)}"
        new.mkdir()
        self._length += 1
