import importlib

import currytruck
from currytruck import CurryTruckError, InvalidHeatLevel, InfeasibleError, UnboundedError, EmptySpaceError


def test_version_matches_pyproject():
    from pathlib import Path

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if pyproject.exists():
        assert f'version = "{currytruck.__version__}"' in pyproject.read_text(encoding="utf-8")


def test_all_exports_resolve():
    for name in currytruck.__all__:
        assert hasattr(currytruck, name), name


def test_modules_are_namespaces():
    for mod in ("spice", "blend", "simmer", "heat", "recipe", "taste", "portion", "balance", "serve"):
        assert getattr(currytruck, mod) is importlib.import_module(f"currytruck.{mod}")


def test_exception_hierarchy():
    for exc in (InvalidHeatLevel, InfeasibleError, UnboundedError, EmptySpaceError):
        assert issubclass(exc, CurryTruckError)
        assert issubclass(exc, ValueError)


def test_result_better_than():
    a = currytruck.Result(x=None, score=1.0)
    b = currytruck.Result(x=None, score=2.0)
    assert a.better_than(b)
    a.maximize = b.maximize = True
    assert b.better_than(a)
