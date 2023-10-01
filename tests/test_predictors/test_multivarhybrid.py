import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.multivarhybrid import HybridMulti

data = pd.read_csv("tests/example_dataset/CaliforniaHousing.csv")
data_small = data[:20]

test0 = HybridMulti(
    1,
    5,
    5,
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
    scale="standard",
)


test1 = HybridMulti(
    1,
    5,
    5,
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
    scale="standard",
)


test1 = HybridMulti(
    1,
    5,
    5,
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
    scale="standard",
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
            [0.98214266],
            [-0.60701891],
            [1.85618152],
            [1.85618152],
            [1.85618152],
            [0.62855945],
            [0.32704136],
            [1.15562047],
            [0.15696608],
            [0.3447108],
            [-0.15375759],
            [-0.26333577],
            [-0.04901636],
            [-0.04983292],
            [-0.03290586],
            [-0.9744286],
            [0.86143887],
            [-0.82077735],
            [-0.76602806],
            [-0.75984669],
            [-0.04959654],
            [-0.09251223],
            [-0.02584253],
            [-0.0503293],
            [-0.08561576],
            [1.05254828],
            [1.04318455],
            [1.03850269],
            [1.03850269],
            [1.03850269],
            [-1.32783522],
            [-1.32284391],
            [-1.33282653],
            [-1.33781784],
            [-1.33781784],
            [2.34476576],
            [2.33223796],
            [1.7826994],
            [0.93296751],
            [-0.012881],
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
