import numpy as np
import unittest
import pandas as pd

from imbrium.predictors.multivarhybrid import *

data = pd.read_csv('tests/example_dataset/CaliforniaHousing.csv')

test0 = HybridMultStepMultVar(
    1,
    5,
    5,
    data=data,
    features=[
        'target',
        'HouseAge',
        'AveRooms',
        'AveBedrms',
        'Population',
        'AveOccup',
        'Latitude',
        'Longitude',
        'MedInc'],
        scale = 'standard')

test0.create_cnnlstm(optimizer  = 'adam',
                     loss  = 'mean_squared_error',
                     metrics  = 'mean_squared_error',
                     layer_config = {'layer0': (64, 1, 'relu'),
                                     'layer1': (32, 1, 'relu'),
                                     'layer2': (2),
                                     'layer3': (50, 'relu'),
                                     'layer4': (25, 'relu')})

X = np.array([[[0.98214266],
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
               [-0.012881]]])

y = np.array([3.422, 2.697, 2.992, 2.414, 2.267])

shape_x = (20632, 1, 40, 1)
shape_y = (20632, 5)

model_id = 'CNN-LSTM'
optimizer = 'adam'
loss = 'mean_squared_error'
metrics = 'mean_squared_error'

class TestHybrid(unittest.TestCase):

    def test_get_model_id(self):
        self.assertEqual(test0.get_model_id, model_id)

    def test_get_X_input(self):
        np.testing.assert_allclose(test0.get_X_input[0], X, rtol=1e-05, atol=0)

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

if __name__ == '__main__':
    unittest.main()
