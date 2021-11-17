import pytest

import dinf


class TestStore:
    @pytest.mark.usefixtures("tmp_path")
    def test_empty(self, tmp_path):
        store = dinf.Store(tmp_path)
        assert len(store) == 0
        with pytest.raises(IndexError):
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

    @pytest.mark.usefixtures("tmp_path")
    def test_base_not_a_directory(self, tmp_path):
        path = tmp_path / "not-a-dir"
        with open(path, "w") as f:
            f.write("blah\n")
        with pytest.raises(ValueError):
            dinf.Store(path)

    @pytest.mark.usefixtures("tmp_path")
    def test_sequence_dir_not_a_directory(self, tmp_path):
        with open(tmp_path / "0", "w") as f:
            f.write("blah\n")
        with pytest.raises(ValueError):
            dinf.Store(tmp_path)

    @pytest.mark.usefixtures("tmp_path")
    def test_str(self, tmp_path):
        base = tmp_path
        store = dinf.Store(base)
        assert str(store) == f"{base}"
        store.increment()
        assert str(store) == f"{base}/0"
        store.increment()
        assert str(store) == f"{base}/" + "{0..1}"
        store.increment()
        assert str(store) == f"{base}/" + "{0..2}"
