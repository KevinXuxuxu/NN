import unittest

import numpy as np

from src.model import Model


class ModelTest(unittest.TestCase):
    def _prepare_test_model(self) -> Model:
        # set large learning rate for more significant result
        model = Model(learning_rate=10).layer(5).layer(3).layer(2)

        # set weights and bias to make test result stable
        model._layers[1]._w = np.array(
            [
                [0.07976754, 0.51339341, 0.92856542],
                [0.30049373, 0.61717666, 0.48434201],
                [0.70701189, 0.82038322, 0.3344529],
                [0.05056967, 0.02322785, 0.53843972],
                [0.0330928, 0.00661952, 0.02314041],
            ]
        )
        model._layers[1]._b = np.array([0.60902404, 0.69674186, 0.41454241])
        model._layers[2]._w = np.array(
            [
                [0.29455487, 0.3540086],
                [0.59675014, 0.30835756],
                [0.38307017, 0.03837065],
            ]
        )
        model._layers[2]._b = np.array([0.32968897, 0.07806679])
        return model

    def _assert_ndarray_almost_equal(self, a: np.ndarray, b: np.ndarray):
        self.assertEqual(a.shape, b.shape)
        flat_n = np.prod(a.shape)
        a_flat, b_flat = a.reshape(flat_n).tolist(), b.reshape(flat_n).tolist()
        for i in range(flat_n):
            self.assertAlmostEqual(a_flat[i], b_flat[i])

    def test_inference(self):
        _input = np.array([[1.0, 0.0, -1.0, 0.0, 1.0]])
        model = self._prepare_test_model()
        model.inference(_input)
        self._assert_ndarray_almost_equal(
            model._layers[1].a, np.array([[0.50371805, 0.59781558, 0.73726381]])
        )
        self._assert_ndarray_almost_equal(
            model._layers[1]._z, np.array([[0.01487249, 0.39637157, 1.03179533]])
        )
        self._assert_ndarray_almost_equal(
            model._layers[2].a, np.array([[0.7534749, 0.6151512]])
        )
        self._assert_ndarray_almost_equal(
            model._layers[2]._z, np.array([[1.11723189, 0.46901756]])
        )

    def test_back_prop(self):
        _input = np.array([[1.0, 0.0, -1.0, 0.0, 1.0]])
        model = self._prepare_test_model()
        model.inference(_input)
        ground_truth = np.array([[1.0, 0.0]])
        model.back_prop(ground_truth)
        self._assert_ndarray_almost_equal(
            model._layers[1]._w,
            np.array(
                [
                    [-0.11055359, 0.42885772, 0.97487527],
                    [0.30049373, 0.61717666, 0.48434201],
                    [0.89733302, 0.90491891, 0.28814305],
                    [0.05056967, 0.02322785, 0.53843972],
                    [-0.15722833, -0.07791617, 0.06945026],
                ]
            ),
        )
        self._assert_ndarray_almost_equal(
            model._layers[1]._b, np.array([0.41870291, 0.61220617, 0.46085226])
        )
        self._assert_ndarray_almost_equal(
            model._layers[2]._w,
            np.array(
                [
                    [0.75588157, -1.11313087],
                    [1.14425541, -1.43285228],
                    [1.05828814, -2.10899895],
                ]
            ),
        )
        self._assert_ndarray_almost_equal(
            model._layers[2]._b, np.array([1.24553207, -2.83455358])
        )


if __name__ == "__main__":
    unittest.main()
