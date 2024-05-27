import numpy as np
import pandas as pd
import pytest

from imbrium.predictors.multivarhybrid import HybridMulti

data = pd.read_csv("tests/example_dataset/mockData.csv")
target = np.array(data["target"])
features = np.array(
    data[
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
        ]
    ]
)
data_small = data[:20]
target_small = np.array(data_small["target"])
features_small = np.array(
    data_small[
        [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
        ]
    ]
)

test0 = HybridMulti(
    target=target,
    features=features,
)


test1 = HybridMulti(
    target=target_small,
    features=features_small,
)


test0.create_cnnlstm(
    sub_seq=1,
    steps_past=5,
    steps_future=5,
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

shape_x = (441, 1, 40, 1)
shape_y = (441, 5)

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
        sub_seq=1,
        steps_past=5,
        steps_future=5,
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
        sub_seq=1,
        steps_past=5,
        steps_future=5,
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
        sub_seq=1,
        steps_past=5,
        steps_future=5,
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
        sub_seq=1,
        steps_past=5,
        steps_future=5,
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
        sub_seq=1,
        steps_past=5,
        steps_future=5,
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
        sub_seq=1,
        steps_past=5,
        steps_future=5,
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


def test_create_fit_cnnrnn_evaluate():
    try:
        test1.create_fit_cnnrnn(sub_seq=1, steps_past=5, steps_future=5, epochs=1)
        test1.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnlstm_evaluate():
    try:
        test1.create_fit_cnnlstm(sub_seq=1, steps_past=5, steps_future=5, epochs=1)
        test1.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnngru_evaluate():
    try:
        test1.create_fit_cnngru(sub_seq=1, steps_past=5, steps_future=5, epochs=1)
        test1.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbirnn_evaluate():
    try:
        test1.create_fit_cnnbirnn(sub_seq=1, steps_past=5, steps_future=5, epochs=1)
        test1.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbilstm_evaluate():
    try:
        test1.create_fit_cnnbilstm(sub_seq=1, steps_past=5, steps_future=5, epochs=1)
        test1.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")


def test_create_fit_cnnbigru_evaluate():
    try:
        test1.create_fit_cnnbigru(sub_seq=1, steps_past=5, steps_future=5, epochs=1)
        test1.evaluate_model()
    except Exception as e:
        pytest.fail(f"An exception was raised: {e}")
