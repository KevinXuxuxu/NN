import numpy as np
import unittest

from src.operator.operator import Operator, ParameterizedOperator
from src.operator.common import Bias, Linear, DataInput
from src.operator.activation import Sigmoid
from src.operator.loss import MSE, CrossEntropy, CrossEntropyWithSoftmax
from src.operator.normalizer import Softmax
from typing import List


class Plus(Operator):

    def __init__(self, p: int):
        super().__init__([0], [0])
        self.p = p
        self.output = None

    def _forward_pass(self):
        self.output = sum(pred.output for pred in self.preds) + self.p
        super()._forward_pass()


class OperatorTest(unittest.TestCase):

    def test_forward_pass_topo_sort(self):
        expected_result = [0, 1, 2, 5, 9, 14, 23, 31, 53]
        a = [Plus(i) for i in range(9)]
        a[1].add_pred(a[0])
        a[2].add_pred(a[0])
        a[6].add_pred(a[1])
        a[7].add_pred(a[1])
        a[6].add_pred(a[2])
        a[3].add_pred(a[2])
        a[6].add_pred(a[3])
        a[4].add_pred(a[3])
        a[6].add_pred(a[4])
        a[5].add_pred(a[4])
        a[8].add_pred(a[5])
        a[7].add_pred(a[6])
        a[8].add_pred(a[7])
        a[0].output = 0
        for succ in a[0].succs:
            succ.forward_pass()
        for i, x in enumerate(a):
            self.assertEqual(x.output, expected_result[i])
            self.assertEqual(x.forward_pass_count, 0)

    def _prepare_test_case(self) -> List[Operator]:
        # _input - Linear1 - Bias1 - Sigmoid1 - Linear2 - Bias2 - Sigmoid2 - MSE
        #                                                                  /
        #                                                    _ground_truth
        ParameterizedOperator.rate = 1.
        _input = DataInput(np.array([[.7, 2.]]))
        _ground_truth = DataInput(np.array([[1., 0.]]))
        mm1 = Linear([2], [3])
        mm1._w = np.array([
            [.1, 2, -.7],
            [-5, .1, 1.5]
        ])
        mm1.add_pred(_input)
        b1 = Bias([3], [3])
        b1._b = np.zeros(3)
        b1.add_pred(mm1)
        a1 = Sigmoid()
        a1.add_pred(b1)
        mm2 = Linear([3], [2])
        mm2._w = np.array([
            [.3, -1.5],
            [1, .2],
            [.2, 3]
        ])
        mm2.add_pred(a1)
        b2 = Bias([2],[2])
        b2._b = np.zeros(2)
        b2.add_pred(mm2)
        a2 = Sigmoid()
        a2.add_pred(b2)
        loss = MSE([2])
        loss.add_ground_truth(_ground_truth)
        _ground_truth.forward_pass()
        loss.add_output(a2)
        return [_input, _ground_truth, mm1, b1, a1, mm2, b2, a2, loss]

    def _assert_matrices_equal(self, a: np.ndarray, b: np.ndarray):
        self.assertEqual(a.shape, b.shape)
        n = np.prod(a.shape)
        # compare flattened
        for i in range(n):
            self.assertAlmostEqual(a.reshape((n))[i], b.reshape((n))[i])

    def test_forward_pass(self):
        # initialize
        _input, _, _, b1, a1, _, b2, a2, _ = self._prepare_test_case()

        # forward pass
        _input.forward_pass()

        # check result
        self._assert_matrices_equal(b1.output, np.array([[-9.93, 1.6, 2.51]]))
        self._assert_matrices_equal(a1.output, np.array([
            [4.8689425323061825e-05, 0.8320183851339245, 0.9248398905178734]]))
        self._assert_matrices_equal(b2.output, np.array([
            [1.0170009700650962, 2.9408503144424207]]))
        self._assert_matrices_equal(a2.output, np.array([
            [0.7343880132771364, 0.949829262885629]]))
    
    def test_back_prop(self):
        # initialize
        _input, _, mm1, b1, _, mm2, b2, _, loss = self._prepare_test_case()
        _input.forward_pass()

        # back prop
        loss._back_prop()

        # check result
        self._assert_matrices_equal(b2._b, np.array(
            [0.10362174841852187, -0.09052563259036336]))
        self._assert_matrices_equal(mm2._w, np.array([
            [0.30000504528338146, -1.500004407641028],
            [1.0862151997839324, 0.12468100935893892],
            [0.29583352646265637, 2.9162782838660672]]))
        self._assert_matrices_equal(b1._b, np.array(
            [8.12465095e-06, 1.19521274e-02, -1.74370284e-02]))
        self._assert_matrices_equal(mm1._w, np.array([
            [0.10000569, 2.00836649, -0.71220592],
            [-4.99998375, 0.12390425, 1.46512594]]))

    def test_cross_entropy_and_softmax(self):
        #        Softmax - CrossEntropy
        #       /          /
        # _input    _ground_truth
        #       \         \
        #        CrossEntropyWithSoftmax
        _input = DataInput(np.array([
            [1, 4, 20],
            [4, 0.2, 7],
            [12, 13, 1.]
        ]))
        _ground_truth = DataInput(np.array([
            [0, 0, 1.],
            [0, 1., 0],
            [1., 0, 0]
        ]))
        sm = Softmax()
        sm.add_pred(_input)
        ce = CrossEntropy([3])
        ce.add_output(sm)
        ce.add_ground_truth(_ground_truth)
        cewsm = CrossEntropyWithSoftmax([3])
        cewsm.add_output(_input)
        cewsm.add_ground_truth(_ground_truth)
        _ground_truth.forward_pass()
        _input.forward_pass()
        ce._back_prop()
        cewsm._back_prop()

        self.assertAlmostEqual(ce.output, cewsm.output)
        self._assert_matrices_equal(sm.grad, cewsm.grad)


if __name__ == '__main__':
    unittest.main()
