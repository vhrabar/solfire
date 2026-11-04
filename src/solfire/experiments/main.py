"""
``solfire-lab``: the ``solfire`` command line with every dropped experiment added.

``solfire-lab train --magnetic-head-weight 0.1 ...``, ``solfire-lab fit-filter ...``, and the
``predict``/``calibrate``/``evaluate`` commands for checkpoints and calibrations of the experiments.
"""

from solfire import main as core

from .extension import EXPERIMENTS


def parse_args(argv=None):
    """
    :param argv: Arguments, or ``None`` for ``sys.argv``
    :return: Parsed arguments, the experiments' options included
    """
    return core.parse_args(argv, EXPERIMENTS)


def build_model_config(args) -> dict:
    """
    :param args: Parsed ``train`` arguments
    :return: Constructor arguments of any architecture, the experiments' included
    """
    return core.build_model_config(args, EXPERIMENTS)


def main(argv=None) -> None:
    """
    Run the command line with the experiments.
    :param argv: Arguments, or ``None`` for ``sys.argv``
    :return: None
    """
    core.main(argv, EXPERIMENTS)


if __name__ == "__main__":
    main()
