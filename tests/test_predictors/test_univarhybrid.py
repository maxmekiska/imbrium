import numpy as np
import unittest
import pandas as pd

from imbrium.predictors.univarhybrid import *

data = pd.read_csv('tests/example_dataset/CaliforniaHousing.csv')
data = data['target']

test0 = HybridMultStepUniVar(
    2,
    10,
    3,
    data=data,
    scale = 'standard')

X = np.array([[[ 1.17289952],
               [ 0.54461086],
               [ 0.80025935],
               [ 0.29936163],
               [ 0.17197069]],
              [[ 0.47008283],
               [ 0.64687025],
               [ 0.30282805],
               [ 0.05757883],
               [-0.13480749]]])

y = np.array([[ 0.05757883],
              [-0.13480749],
              [-0.41298771]])


shape_x = (20628, 2, 5, 1)
shape_y = (20628, 3, 1)


class TestUnivarhybrid(unittest.TestCase):

    def test_get_X_input(self):
        np.testing.assert_allclose(test0.get_X_input[4], X)

    def test_get_y_input(self):
        np.testing.assert_allclose(test0.get_y_input[2], y)

    def test_get_X_input_shape(self):
        np.testing.assert_allclose(test0.get_X_input_shape, shape_x)

    def test_get_y_input_shape(self):
        np.testing.assert_allclose(test0.get_y_input_shape, shape_y)


if __name__ == '__main__':
    unittest.main()
