import pytest

import dinf

class TestStore:
    @pytest.mark.usefixtures("tmp_path")
    def test_empty(self, tmp_path):
        store = dinf.Store(tmp_path)
        assert len(store) == 0
        with pytest.raises(KeyError):
            store[-1]

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.parametrize("size", [1, 5, 300])
    def test_nonempty(self, tmp_path, size):
        for i in range(size):
            (tmp_path / f"{i}").mkdir()
        store = dinf.Store(tmp_path)
        assert len(store) == size
        assert store[-1].exists()
        assert not (tmp_path / f"{size}").exists()

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.parametrize("size", [1, 5, 300])
    def test_increment(self, tmp_path, size):
        store = dinf.Store(tmp_path)
        assert len(store) == 0

        for i in range(size):
            assert len(store) == i
            store.increment()
            assert store[-1].exists()
        assert len(store) == size
        for i in range(size):
            assert (tmp_path / f"{i}").exists()
        assert not (tmp_path / f"{size}").exists()

    @pytest.mark.usefixtures("tmp_path")
    @pytest.mark.parametrize("size", [1, 5, 300])
    def test_getitem(self, tmp_path, size):
        for i in range(size):
            (tmp_path / f"{i}").mkdir()
        store = dinf.Store(tmp_path)
        assert len(store) == size
        assert store[-1] == tmp_path / f"{size - 1}"
        for i in range(size):
            assert store[i] == (tmp_path / f"{i}")

    @pytest.mark.usefixtures("tmp_path")
    def test_getitem_badkey(self, tmp_path):
        store = dinf.Store(tmp_path)
        with pytest.raises(TypeError):
            store["0"]
        with pytest.raises(TypeError):
            store[0:1]
