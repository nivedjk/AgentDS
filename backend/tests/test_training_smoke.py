"""Smoke test: the Module 5 training path can import its ML dependencies.

The training/AutoML path is fully deterministic and has no LLM anywhere in it.
These tests just prove the wheels are installed and importable; the optional
boosters skip cleanly if a wheel is missing.
"""

import sklearn
import joblib
import pytest


def test_core_ml_deps_import():
    assert isinstance(sklearn.__version__, str) and sklearn.__version__
    assert isinstance(joblib.__version__, str) and joblib.__version__


def test_optional_boosters_import():
    xgboost = pytest.importorskip("xgboost")
    lightgbm = pytest.importorskip("lightgbm")
    assert isinstance(xgboost.__version__, str) and xgboost.__version__
    assert isinstance(lightgbm.__version__, str) and lightgbm.__version__
