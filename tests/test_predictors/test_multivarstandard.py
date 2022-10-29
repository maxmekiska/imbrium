import numpy as np
import unittest
import pandas as pd

from imbrium.predictors.multivarstandard import *

data = pd.read_csv("tests/example_dataset/CaliforniaHousing.csv")

test0 = BasicMultStepMultVar(
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
    scale="standard",
)

test0.create_lstm(
    optimizer="adam",
    loss="mean_squared_error",
    metrics="mean_squared_error",
    layer_config={
        "layer0": (40, "relu"),
        "layer1": (50, "relu"),
        "layer2": (50, "relu"),
    },
)

X = np.array(
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

y = np.array([3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 2.414, 2.267, 2.611, 2.815])

shape_x = (20630, 2, 8)
shape_y = (20630, 10)

model_id = "LSTM"
optimizer = "adam"
loss = "mean_squared_error"
metrics = "mean_squared_error"


class TestStandard(unittest.TestCase):
    def test_get_model_id(self):
        self.assertEqual(test0.get_model_id, model_id)

    def test_get_X_input(self):
        np.testing.assert_allclose(test0.get_X_input[0], X)

    def test_get_y_input(self):
        np.testing.assert_allclose(test0.get_y_input[0], y)

    def test_get_X_input_shape(self):
        np.testing.assert_allclose(test0.get_X_input_shape, shape_x)

    def test_get_y_input_shape(self):
        np.testing.assert_allclose(test0.get_y_input_shape, shape_y)

    def test_get_optimizer(self):
        self.assertEqual(test0.get_optimizer, optimizer)

    def test_get_loss(self):
        self.assertEqual(test0.get_loss, loss)

    def test_get_metrics(self):
        self.assertEqual(test0.get_metrics, metrics)


if __name__ == "__main__":
    unittest.main()
