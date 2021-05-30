import unittest
import numpy as np

from typing import List
from src.layer import Layer


class LayerTest(unittest.TestCase):

    def _prepare_test_case(self) -> List[Layer]:
        learning_rate = 1.
        layers = [
            Layer(2, learning_rate),
            Layer(3, learning_rate),
            Layer(2, learning_rate)
        ]
        layers[2].connect(layers[1])
        layers[1].connect(layers[0])
        # initialization
        layers[0].a = np.array([[.7, 2.]]).transpose()
        layers[1]._b = np.zeros((3, 1))
        layers[1]._w = np.array(
            [
                [.1, -5],
                [2., .1],
                [-.7, 1.5]
            ]
        )
        layers[2]._b = np.zeros((2, 1))
        layers[2]._w = np.array(
            [
                [.3, 1., .2],
                [-1.5, .2, 3.]
            ]
        )
        return layers

    def test_forward_pass(self):
        layers = self._prepare_test_case()
        layers[1].forward_pass()
        self.assertAlmostEqual(layers[1]._z[0][0], -9.93)
        self.assertAlmostEqual(layers[1]._z[1][0], 1.6)
        self.assertAlmostEqual(layers[1]._z[2][0], 2.51)
        self.assertAlmostEqual(layers[1].a[0][0], 4.8689425323061825e-05)
        self.assertAlmostEqual(layers[1].a[1][0], 0.8320183851339245)
        self.assertAlmostEqual(layers[1].a[2][0], 0.9248398905178734)
        self.assertAlmostEqual(layers[2]._z[0][0], 1.0170009700650962)
        self.assertAlmostEqual(layers[2]._z[1][0], 2.9408503144424207)
        self.assertAlmostEqual(layers[2].a[0][0], 0.7343880132771364)
        self.assertAlmostEqual(layers[2].a[1][0], 0.949829262885629)

    def test_back_prop(self):
        layers = self._prepare_test_case()
        layers[1].forward_pass()
        a_grad = 2 * (layers[-1].a - np.array([[1., 0.]]).transpose())
        layers[2].back_prop(a_grad)
        self.assertAlmostEqual(layers[2]._b[0][0], 0.10362174841852187)
        self.assertAlmostEqual(layers[2]._b[1][0], -0.09052563259036336)
        self.assertAlmostEqual(layers[2]._w[0][0], 0.30000504528338146)
        self.assertAlmostEqual(layers[2]._w[0][1], 1.0862151997839324)
        self.assertAlmostEqual(layers[2]._w[0][2], 0.29583352646265637)
        self.assertAlmostEqual(layers[2]._w[1][0], -1.500004407641028)
        self.assertAlmostEqual(layers[2]._w[1][1], 0.12468100935893892)
        self.assertAlmostEqual(layers[2]._w[1][2], 2.9162782838660672)
        # TODO: fill out the last layer's _w and _b assertion


if __name__ == '__main__':
    unittest.main()