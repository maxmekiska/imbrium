import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.multivarhybrid import HybridMulti

data = pd.read_csv("tests/example_dataset/CaliforniaHousing.csv")
target = np.array(data["target"])
features = np.array(
    data[
        [
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "MedInc",
        ]
    ]
)
data_small = data[:20]
target_small = np.array(data_small["target"])
features_small = np.array(
    data_small[
        [
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
            "MedInc",
        ]
    ]
)

test0 = HybridMulti(
    1,
    5,
    5,
    # data=data,
    # features=[
    # "target",
    # "HouseAge",
    # "AveRooms",
    # "AveBedrms",
    # "Population",
    # "AveOccup",
    # "Latitude",
    # "Longitude",
    # "MedInc",
    # ],
    target=target,
    features=features,
)


test1 = HybridMulti(
    1,
    5,
    5,
    # data=data_small,
    # features=[
    # "target",
    # "HouseAge",
    # "AveRooms",
    # "AveBedrms",
    # "Population",
    # "AveOccup",
    # "Latitude",
    # "Longitude",
    # "MedInc",
    # ],
    target=target_small,
    features=features_small,
)


test1 = HybridMulti(
    1,
    5,
    5,
    # data=data_small,
    # features=[
    # "target",
    # "HouseAge",
    # "AveRooms",
    # "AveBedrms",
    # "Population",
    # "AveOccup",
    # "Latitude",
    # "Longitude",
    # "MedInc",
    # ],
    target=target_small,
    features=features_small,
)

test0.create_cnnlstm(
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
        "layer4": {
            "config": {
                "neurons": 32,
                "activation": "relu",
                "regularization": 0.0,
            }
        },
    },
)


X = np.array(
    [
        [
            [4.10000000e01],
            [2.10000000e01],
            [5.20000000e01],
            [5.20000000e01],
            [5.20000000e01],
            [6.98412698e00],
            [6.23813708e00],
            [8.28813559e00],
            [5.81735160e00],
            [6.28185328e00],
            [1.02380952e00],
            [9.71880492e-01],
            [1.07344633e00],
            [1.07305936e00],
            [1.08108108e00],
            [3.22000000e02],
            [2.40100000e03],
            [4.96000000e02],
            [5.58000000e02],
            [5.65000000e02],
            [2.55555556e00],
            [2.10984183e00],
            [2.80225989e00],
            [2.54794520e00],
            [2.18146718e00],
            [3.78800000e01],
            [3.78600000e01],
            [3.78500000e01],
            [3.78500000e01],
            [3.78500000e01],
            [-1.22230000e02],
            [-1.22220000e02],
            [-1.22240000e02],
            [-1.22250000e02],
            [-1.22250000e02],
            [8.32520000e00],
            [8.30140000e00],
            [7.25740000e00],
            [5.64310000e00],
            [3.84620000e00],
        ]
    ]
)

y = np.array([3.422, 2.697, 2.992, 2.414, 2.267])

shape_x = (20632, 1, 40, 1)
shape_y = (20632, 5)

model_id = "CNN-LSTM"
optimizer = "adam"
loss = "mean_squared_error"
metrics = "mean_squared_error"


def test_get_model_id():
    assert test0.get_model_id == model_id


def test_get_X_input():
    np.testing.assert_allclose(test0.get_X_input[0], X, rtol=1e-05, atol=0)


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


def test_create_cnnrnn():
    test0.create_cnnrnn(
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
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test0.get_model_id == "CNN-RNN"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnlstm():
    test0.create_cnnlstm(
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
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test0.get_model_id == "CNN-LSTM"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnngru():
    test0.create_cnngru(
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
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test0.get_model_id == "CNN-GRU"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbirnn():
    test0.create_cnnbirnn(
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
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test0.get_model_id == "CNN-BI-RNN"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbilstm():
    test0.create_cnnbilstm(
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
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test0.get_model_id == "CNN-BI-LSTM"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_cnnbigru():
    test0.create_cnnbigru(
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
            "layer4": {
                "config": {
                    "neurons": 32,
                    "activation": "relu",
                    "regularization": 0.0,
                }
            },
        },
    )
    assert test0.get_model_id == "CNN-BI-GRU"
    assert test0.get_optimizer == "adam"
    assert test0.get_loss == "mean_squared_error"
    assert test0.get_metrics == "mean_squared_error"


def test_create_fit_cnnrnn():
    try:
        test1.create_fit_cnnrnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnlstm():
    try:
        test1.create_fit_cnnlstm(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_cnngru():
    try:
        test1.create_fit_cnngru(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbirnn():
    try:
        test1.create_fit_cnnbirnn(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbilstm():
    try:
        test1.create_fit_cnnbilstm(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbigru():
    try:
        test1.create_fit_cnnbigru(epochs=1)
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")
