import contextlib
import io
import os

# test on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"


@contextlib.contextmanager
def capture():
    """
    Context manager for capturing stdout, stderr and exit code.
    """

    class _Capture:
        pass

    cap = _Capture()
    with contextlib.redirect_stdout(io.StringIO()) as fout:
        with contextlib.redirect_stderr(io.StringIO()) as ferr:
            try:
                yield cap
                cap.ret = 0
            except SystemExit as e:
                cap.ret = e.code
            finally:
                cap.out = fout.getvalue()
                cap.err = ferr.getvalue()
