"""
The ``solfire`` command line is the final method alone: no experiment reaches it.
"""

import subprocess
import sys

import pytest

from solfire.extension import Extension
from solfire.main import parse_args


def run_isolated(code: str) -> str:
    """
    Run ``code`` in a fresh interpreter, so no other test has imported the experiments yet.
    :param code: Python source
    :return: Its standard output
    """
    return subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.strip()


def test_core_never_imports_the_experiments():
    out = run_isolated(
        "import sys, solfire.main, solfire.inference, solfire.training; "
        "print(any(name.startswith('solfire.experiments') for name in sys.modules))"
    )
    assert out == "False"


def test_core_trains_only_the_pretrained_unets():
    out = run_isolated(
        "from solfire.networks import ARCHITECTURES; from solfire.data import preprocessing_methods; "
        "print(sorted(ARCHITECTURES), preprocessing_methods())"
    )
    assert out == "['pretrained'] ('none',)"
    args = parse_args(["train"])
    assert args.arch == "pretrained" and args.preprocess == "none"


@pytest.mark.parametrize("option", ["--magnetic-head-weight", "--morph-open-weight", "--pseudo-labels", "--tversky-beta"])
def test_experiment_options_are_not_core_options(option, capsys):
    with pytest.raises(SystemExit):
        parse_args(["train", option, "0.1"])
    assert "unrecognized arguments" in capsys.readouterr().err


def test_core_refuses_a_learned_filter_calibration():
    assert Extension().instance_filter({"threshold": 0.5}) is None
    with pytest.raises(SystemExit, match="solfire-lab"):
        Extension().instance_filter({"filter": "runs/x.filter.pkl"})
