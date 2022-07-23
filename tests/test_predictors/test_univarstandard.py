import numpy as np
import unittest
import pandas as pd

from imbrium.predictors.univarstandard import *

data = pd.read_csv('tests/example_dataset/CaliforniaHousing.csv')
data = data['target']

test0 = BasicMultStepUniVar(
    2,
    3,
    data=data,
    scale = 'standard')

test0.create_lstm(optimizer = 'adam',
                  loss = 'mean_squared_error',
                  metrics = 'mean_squared_error',
                  layer_config = {'layer0': (40, 'relu'),
                                  'layer1': (50,'relu'),
                                  'layer2': (50, 'relu')})

X = np.array([[1.17289952],
              [0.54461086]])

y = np.array([[1.17289952],
              [0.54461086],
              [0.80025935]])

shape_x = (20636, 2, 1)
shape_y = (20636, 3, 1)

loss = 'mean_squared_error'
metrics = 'mean_squared_error'

class TestUnivarstandard(unittest.TestCase):

    def test_get_X_input(self):
        np.testing.assert_allclose(test0.get_X_input[4], X)

    def test_get_y_input(self):
        np.testing.assert_allclose(test0.get_y_input[2], y)

    def test_get_X_input_shape(self):
        np.testing.assert_allclose(test0.get_X_input_shape, shape_x)

    def test_get_y_input_shape(self):
        np.testing.assert_allclose(test0.get_y_input_shape, shape_y)

    def test_get_loss(self):
        self.assertEqual(test0.get_loss, loss)

    def test_get_metrics(self):
        self.assertEqual(test0.get_metrics, metrics)

if __name__ == '__main__':
    unittest.main()
