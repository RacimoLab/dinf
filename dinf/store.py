import collections.abc
import dataclasses
import pathlib


class Store(collections.abc.Sequence):
    base: pathlib.Path

    def __init__(self, base):
        self.base = pathlib.Path(base)
        if not self.base.exists():
            self.base.mkdir(parents=True)
        if not self.base.is_dir():
            raise ValueError("{self.base} is not a directory")

        # Find the length.
        self._length = 0
        while True:
            try:
                self[self._length]
            except KeyError:
                break
            self._length += 1

    def __len__(self):
        return self._length

    def __getitem__(self, key):
        if isinstance(key, int):
            if key < 0:
                key += len(self)
            path = self.base / f"{key}"
            if not path.exists():
                raise KeyError
            return path
        else:
            raise TypeError("key must be an integer, but got {type(key)}")

    def increment(self):
        new = self.base / f"{len(self)}"
        new.mkdir()
        self._length += 1
