from warnings import filterwarnings

import numpy as np
import pandas as pd
import pytest

from imbrium.utils.scaler import SCALER
from imbrium.utils.transformer import *

data_uni = {"test0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
df_uni = pd.DataFrame(data=data_uni)

data_multi = {
    "test0": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "test1": [2, 23, 1, 20, 60, 90, 30, 10, 20, 300],
    "test2": [1, 5, 300, 600, 100, 200, 400, 60, 100, 6000],
}

df_multi = pd.DataFrame(data=data_multi)

df_uni_output = np.array(
    [
        [-1.5666989],
        [-1.21854359],
        [-0.87038828],
        [-0.52223297],
        [-0.17407766],
        [0.17407766],
        [0.52223297],
        [0.87038828],
        [1.21854359],
        [1.5666989],
    ]
)

df_multi_output = np.array(
    [
        [
            2.00000000e00,
            2.30000000e01,
            1.00000000e00,
            2.00000000e01,
            6.00000000e01,
            9.00000000e01,
            3.00000000e01,
            1.00000000e01,
            2.00000000e01,
            3.00000000e02,
        ],
        [
            -4.43073061e-01,
            -4.40788002e-01,
            -2.72264854e-01,
            -1.00885382e-01,
            -3.86517836e-01,
            -3.29391345e-01,
            -2.15138364e-01,
            -4.09368432e-01,
            -3.86517836e-01,
            2.98394511e00,
        ],
    ]
)

df_sequence_prep_standard_uni_X = np.array(
    [[[-1.5666989], [-1.21854359], [-0.87038828], [-0.52223297], [-0.17407766]]]
)

df_sequence_prep_standard_uni_y = np.array(
    [[[0.17407766], [0.52223297], [0.87038828], [1.21854359], [1.5666989]]]
)

df_sequence_prep_standard_multi_X = np.array(
    [
        [[-1.5666989], [-1.21854359]],
        [[-1.21854359], [-0.87038828]],
        [[-0.87038828], [-0.52223297]],
        [[-0.52223297], [-0.17407766]],
        [[-0.17407766], [0.17407766]],
    ]
)

df_sequence_prep_standard_multi_y = np.array(
    [
        [[-1.21854359], [-0.87038828], [-0.52223297], [-0.17407766], [0.17407766]],
        [[-0.87038828], [-0.52223297], [-0.17407766], [0.17407766], [0.52223297]],
        [[-0.52223297], [-0.17407766], [0.17407766], [0.52223297], [0.87038828]],
        [[-0.17407766], [0.17407766], [0.52223297], [0.87038828], [1.21854359]],
        [[0.17407766], [0.52223297], [0.87038828], [1.21854359], [1.5666989]],
    ]
)


sequence_prep_hybrid_uni_X = np.array(
    [
        [[[-1.5666989]], [[-1.21854359]]],
        [[[-1.21854359]], [[-0.87038828]]],
        [[[-0.87038828]], [[-0.52223297]]],
        [[[-0.52223297]], [[-0.17407766]]],
        [[[-0.17407766]], [[0.17407766]]],
        [[[0.17407766]], [[0.52223297]]],
        [[[0.52223297]], [[0.87038828]]],
    ]
)

sequence_prep_hybrid_uni_y = np.array(
    [
        [[-0.87038828], [-0.52223297]],
        [[-0.52223297], [-0.17407766]],
        [[-0.17407766], [0.17407766]],
        [[0.17407766], [0.52223297]],
        [[0.52223297], [0.87038828]],
        [[0.87038828], [1.21854359]],
        [[1.21854359], [1.5666989]],
    ]
)

sequence_prep_hybrid_multi_X = np.array(
    [
        [[[-1.5666989]], [[-1.21854359]]],
        [[[-1.21854359]], [[-0.87038828]]],
        [[[-0.87038828]], [[-0.52223297]]],
        [[[-0.52223297]], [[-0.17407766]]],
        [[[-0.17407766]], [[0.17407766]]],
        [[[0.17407766]], [[0.52223297]]],
        [[[0.52223297]], [[0.87038828]]],
        [[[0.87038828]], [[1.21854359]]],
    ]
)

sequence_prep_hybrid_multi_y = np.array(
    [
        [[-1.21854359], [-0.87038828]],
        [[-0.87038828], [-0.52223297]],
        [[-0.52223297], [-0.17407766]],
        [[-0.17407766], [0.17407766]],
        [[0.17407766], [0.52223297]],
        [[0.52223297], [0.87038828]],
        [[0.87038828], [1.21854359]],
        [[1.21854359], [1.5666989]],
    ]
)

sequence_prep_hybrid_mod = 1

multistep_prep_standard_X = np.array(
    [
        [[-0.44307306]],
        [[-0.440788]],
        [[-0.27226485]],
        [[-0.10088538]],
        [[-0.38651784]],
        [[-0.32939134]],
        [[-0.21513836]],
        [[-0.40936843]],
        [[-0.38651784]],
    ]
)

multistep_prep_standard_y = np.array(
    [
        [2.0, 23.0],
        [23.0, 1.0],
        [1.0, 20.0],
        [20.0, 60.0],
        [60.0, 90.0],
        [90.0, 30.0],
        [30.0, 10.0],
        [10.0, 20.0],
        [20.0, 300.0],
    ]
)


multistep_prep_hybrid_X = np.array(
    [
        [[[-0.44307306]], [[-0.440788]]],
        [[[-0.440788]], [[-0.27226485]]],
        [[[-0.27226485]], [[-0.10088538]]],
        [[[-0.10088538]], [[-0.38651784]]],
        [[[-0.38651784]], [[-0.32939134]]],
        [[[-0.32939134]], [[-0.21513836]]],
        [[[-0.21513836]], [[-0.40936843]]],
        [[[-0.40936843]], [[-0.38651784]]],
    ]
)

multistep_prep_hybrid_y = np.array(
    [
        [23.0, 1.0],
        [1.0, 20.0],
        [20.0, 60.0],
        [60.0, 90.0],
        [90.0, 30.0],
        [30.0, 10.0],
        [10.0, 20.0],
        [20.0, 300.0],
    ]
)


scaler = SCALER["standard"]


def test_data_prep_uni():
    np.testing.assert_allclose(data_prep_uni(df_uni, scaler), df_uni_output, rtol=1e-07)


def test_data_prep_multi():
    np.testing.assert_allclose(
        data_prep_multi(df_multi, ["test1", "test2"], scaler),
        df_multi_output,
        rtol=1e-07,
    )


def test_sequence_prep_standard_uni_X():
    np.testing.assert_allclose(
        sequence_prep_standard_uni(df_uni_output, 5, 5)[0],
        df_sequence_prep_standard_uni_X,
        rtol=1e-07,
    )


def test_sequence_prep_standard_uni_y():
    np.testing.assert_allclose(
        sequence_prep_standard_uni(df_uni_output, 5, 5)[1],
        df_sequence_prep_standard_uni_y,
        rtol=1e-07,
    )


def test_sequence_prep_standard_multi_X():
    np.testing.assert_allclose(
        sequence_prep_standard_multi(df_uni_output, 2, 5)[0],
        df_sequence_prep_standard_multi_X,
        rtol=1e-07,
    )


def test_sequence_prep_standard_multi_y():
    np.testing.assert_allclose(
        sequence_prep_standard_multi(df_uni_output, 2, 5)[1],
        df_sequence_prep_standard_multi_y,
        rtol=1e-07,
    )


def test_sequence_prep_hybrid_uni_X():
    np.testing.assert_allclose(
        sequence_prep_hybrid_uni(df_uni_output, 2, 2, 2)[0],
        sequence_prep_hybrid_uni_X,
        rtol=1e-07,
    )


def test_sequence_prep_hybrid_uni_y():
    np.testing.assert_allclose(
        sequence_prep_hybrid_uni(df_uni_output, 2, 2, 2)[1],
        sequence_prep_hybrid_uni_y,
        rtol=1e-07,
    )


def test_sequence_prep_hybrid_uni_mod():
    np.testing.assert_allclose(
        sequence_prep_hybrid_uni(df_uni_output, 2, 2, 2)[2],
        sequence_prep_hybrid_mod,
        rtol=1e-07,
    )


def test_sequence_prep_hybrid_multi_X():
    np.testing.assert_allclose(
        sequence_prep_hybrid_multi(df_uni_output, 2, 2, 2)[0],
        sequence_prep_hybrid_multi_X,
        rtol=1e-07,
    )


def test_sequence_prep_hybrid_multi_y():
    np.testing.assert_allclose(
        sequence_prep_hybrid_multi(df_uni_output, 2, 2, 2)[1],
        sequence_prep_hybrid_multi_y,
        rtol=1e-07,
    )


def test_sequence_prep_hybrid_multi_mod():
    np.testing.assert_allclose(
        sequence_prep_hybrid_multi(df_uni_output, 2, 2, 2)[2],
        sequence_prep_hybrid_mod,
        rtol=1e-07,
    )


def test_multistep_prep_standard_X():
    np.testing.assert_allclose(
        multistep_prep_standard(df_multi_output, 1, 2)[0],
        multistep_prep_standard_X,
        rtol=1e-07,
    )


def test_multistep_prep_standard_y():
    np.testing.assert_allclose(
        multistep_prep_standard(df_multi_output, 1, 2)[1],
        multistep_prep_standard_y,
        rtol=1e-07,
    )


def test_multistep_prep_hybrid_X():
    np.testing.assert_allclose(
        multistep_prep_hybrid(df_multi_output, 2, 2, 2)[0],
        multistep_prep_hybrid_X,
        rtol=1e-07,
    )


def test_multistep_prep_hybrid_y():
    np.testing.assert_allclose(
        multistep_prep_hybrid(df_multi_output, 2, 2, 2)[1],
        multistep_prep_hybrid_y,
        rtol=1e-07,
    )


def test_multistep_prep_hybrid_mod():
    np.testing.assert_allclose(
        multistep_prep_hybrid(df_multi_output, 2, 2, 2)[2],
        sequence_prep_hybrid_mod,
        rtol=1e-07,
    )
