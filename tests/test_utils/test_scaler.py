import numpy as np
import pytest

from imbrium.utils.scaler import SCALER

data = np.array(
    [
        [
            0.98214266,
            0.62855945,
            -0.15375759,
            -0.9744286,
            -0.04959654,
            1.05254828,
            -1.32783522,
            2.34476576,
        ],
        [
            -0.60701891,
            0.32704136,
            -0.26333577,
            0.86143887,
            -0.09251223,
            1.04318455,
            -1.32284391,
            2.33223796,
        ],
    ]
)

none_ = np.array(
    [
        [
            0.98214266,
            0.62855945,
            -0.15375759,
            -0.9744286,
            -0.04959654,
            1.05254828,
            -1.32783522,
            2.34476576,
        ],
        [
            -0.60701891,
            0.32704136,
            -0.26333577,
            0.86143887,
            -0.09251223,
            1.04318455,
            -1.32284391,
            2.33223796,
        ],
    ]
)

standard_ = np.array(
    [
        [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0],
    ]
)

minmax_ = np.array(
    [[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]
)


maxabs_ = np.array(
    [
        [1.0, 1.0, -0.58388418, -1.0, -0.53610793, 1.0, -1.0, 1.0],
        [
            -0.61805574,
            0.52030299,
            -1.0,
            0.88404514,
            -1.0,
            0.99110375,
            -0.99624102,
            0.99465712,
        ],
    ]
)

normalize_ = np.array(
    [
        [0.628976, 0.53270003, 0.3196856, 0.09622788, 0.34804725, 0.64814651, 0.0, 1.0],
        [
            0.19626862,
            0.4506007,
            0.28984893,
            0.59610998,
            0.33636189,
            0.64559689,
            0.00135907,
            0.99658885,
        ],
    ]
)


none = SCALER[""].fit(data)
standard = SCALER["standard"].fit(data)
minmax = SCALER["minmax"].fit(data)
maxabs = SCALER["maxabs"].fit(data)
normalize = SCALER["normalize"].fit(data)


def test_none():
    np.testing.assert_allclose(none.transform(data), none_, rtol=1e-04)


def test_standard():
    np.testing.assert_allclose(standard.transform(data), standard_, rtol=1e-04)


def test_minmax():
    np.testing.assert_allclose(minmax.transform(data), minmax_, rtol=1e-04)


def test_maxabs():
    np.testing.assert_allclose(maxabs.transform(data), maxabs_, rtol=1e-04)


def test_normalize():
    np.testing.assert_allclose(normalize.transform(data), normalize_, rtol=1e-04)
