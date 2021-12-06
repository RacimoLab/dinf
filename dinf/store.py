import collections.abc
import pathlib


class Store(collections.abc.Sequence):
    """A sequence of folders with names "0", "1", ..."""

    base: pathlib.Path
    """Base directory containing the sequence."""

    def __init__(self, base):
        self.base = pathlib.Path(base)
        if not self.base.exists():
            self.base.mkdir(parents=True)
        if not self.base.is_dir():
            raise ValueError(f"{self.base} is not a directory")

        # Find the length.
        self._length = 0
        try:
            while True:
                path = self[self._length]
                if not path.is_dir():
                    raise ValueError(f"{path} is not a directory")
                self._length += 1
        except IndexError:
            pass

    def __len__(self):
        return self._length

    def __getitem__(self, index):
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
        new = self.base / f"{len(self)}"
        new.mkdir()
        self._length += 1
