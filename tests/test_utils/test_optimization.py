import pytest

from imbrium.utils.optimization import seeker


def test_seeker():
    def test_func(*args, **kwargs):
        return 1

    test_decorator = seeker(
        optimizer_range=["adam"],
        layer_config_range=[1],
        optimization_target="minimize",
        n_trials=1,
    )

    test_wrapper = test_decorator(test_func)

    test_result = test_wrapper()

    assert test_result == 1
