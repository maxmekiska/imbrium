import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.multivarpure import PureMulti

data = pd.read_csv("tests/example_dataset/CaliforniaHousing.csv")
data_small = data[:20]

test0 = PureMulti(
    2,
    10,
    data=data,
    features=[
        "target",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedInc",
    ],
)

test1 = PureMulti(
    2,
    10,
    data=data_small,
    features=[
        "target",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedInc",
    ],
)


test2 = PureMulti(
    2,
    10,
    data=data_small,
    features=[
        "target",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedInc",
    ],
)


test0.create_lstm(
    optimizer="adam",
    loss="mean_squared_error",
    metrics="mean_squared_error",
    layer_config={
        "layer0": {
            "config": {
                "neurons": 50,
                "activation": "relu",
                "regularization": 0.0,
                "dropout": 0.0,
            }
        },
        "layer1": {
            "config": {
                "neurons": 50,
                "activation": "relu",
                "regularization": 0.0,
                "dropout": 0.0,
            }
        },
        "layer2": {
            "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
        },
    },
)


X = np.array(
    [
        [
            4.10000000e01,
            6.98412698e00,
            1.02380952e00,
            3.22000000e02,
            2.55555556e00,
            3.78800000e01,
            -1.22230000e02,
            8.32520000e00,
        ],
        [
            2.10000000e01,
            6.23813708e00,
            9.71880492e-01,
            2.40100000e03,
            2.10984183e00,
            3.78600000e01,
            -1.22220000e02,
            8.30140000e00,
        ],
    ]
)

y = np.array([3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 2.414, 2.267, 2.611, 2.815])

shape_x = (20630, 2, 8)
shape_y = (20630, 10)

model_id = "LSTM"
optimizer = "adam"
loss = "mean_squared_error"
metrics = "mean_squared_error"


def test_get_model_id():
    assert test0.get_model_id == model_id


def test_get_X_input():
    np.testing.assert_allclose(test0.get_X_input[0], X)


def test_get_y_input():
    np.testing.assert_allclose(test0.get_y_input[0], y)


def test_get_X_input_shape():
    np.testing.assert_allclose(test0.get_X_input_shape, shape_x)


def test_get_y_input_shape():
    np.testing.assert_allclose(test0.get_y_input_shape, shape_y)


def test_get_optimizer():
    assert test0.get_optimizer == optimizer


def test_get_loss():
    assert test0.get_loss == loss


def test_get_metrics():
    assert test0.get_metrics == metrics


def test_create_mlp():
    test0.create_mlp(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
    )
    assert test0.get_model_id == "MLP"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_rnn():
    test1.create_rnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
    )
    assert test1.get_model_id == "RNN"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_lstm():
    test1.create_lstm(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {"neurons": 50, "activation": "relu", "regularization": 0.0}
            },
        },
    )
    assert test1.get_model_id == "LSTM"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_cnn():
    test1.create_cnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": {
                "config": {
                    "filters": 64,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "filters": 32,
                    "kernel_size": 1,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "pool_size": 2,
                }
            },
            "layer3": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
        },
    )
    assert test1.get_model_id == "CNN"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_gru():
    test1.create_gru(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer2": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test1.get_model_id == "GRU"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_birnn():
    test1.create_birnn(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test1.get_model_id == "BI-RNN"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_bilstm():
    test1.create_bilstm(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test1.get_model_id == "BI-LSTM"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_bigru():
    test1.create_bigru(
        optimizer="adam",
        loss="mean_squared_error",
        metrics="mean_squared_error",
        layer_config={
            "layer0": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                    "dropout": 0.0,
                }
            },
            "layer1": {
                "config": {
                    "neurons": 50,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test1.get_model_id == "BI-GRU"
    assert test1.get_optimizer == "adam"
    assert test1.get_loss == "mean_squared_error"
    assert test1.get_metrics == "mean_squared_error"


def test_create_fit_mlp():
    try:
        test1.create_fit_mlp(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_rnn():
    try:
        test2.create_fit_rnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_lstm():
    try:
        test2.create_fit_lstm(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnn():
    try:
        test2.create_fit_cnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_gru():
    try:
        test2.create_fit_gru(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_cerate_fit_birnn():
    try:
        test2.create_fit_birnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_creaste_fit_bilstm():
    try:
        test2.create_fit_bilstm(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def create_fit_bigru():
    try:
        test2.create_fit_bigru(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")
