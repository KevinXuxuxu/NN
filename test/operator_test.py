import numpy as np
import unittest

from src.operator.operator import Operator, ParameterizedOperator
from src.operator.common import Bias, MatMul
from src.operator.activation import Sigmoid
from src.operator.loss import MSE
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
        a[1].register(a[0])
        a[2].register(a[0])
        a[6].register(a[1])
        a[7].register(a[1])
        a[6].register(a[2])
        a[3].register(a[2])
        a[6].register(a[3])
        a[4].register(a[3])
        a[6].register(a[4])
        a[5].register(a[4])
        a[8].register(a[5])
        a[7].register(a[6])
        a[8].register(a[7])
        a[0].output = 0
        for succ in a[0].succs:
            succ.forward_pass()
        for i, x in enumerate(a):
            self.assertEqual(x.output, expected_result[i])
            self.assertEqual(x.forward_pass_count, 0)

    def _prepare_test_case(self) -> List[Operator]:
        # _input - MatMul1 - Bias1 - Sigmoid1 - MatMul2 - Bias2 - Sigmoid2 - MSE
        #                                                                  /
        #                                                    _ground_truth
        ParameterizedOperator.rate = 1.
        _input = Operator([], [2])
        _input.output = np.array([[.7, 2.]])
        _ground_truth = Operator([], [2])
        _ground_truth.output = np.array([[1., 0.]])
        mm1 = MatMul([2], [3])
        mm1._w = np.array(
            [
                [.1, 2, -.7],
                [-5, .1, 1.5]
            ]
        )
        mm1.register(_input)
        b1 = Bias([3], [3])
        b1._b = np.zeros(3)
        b1.register(mm1)
        a1 = Sigmoid()
        a1.register(b1)
        mm2 = MatMul([3], [2])
        mm2._w = np.array(
            [
                [.3, -1.5],
                [1, .2],
                [.2, 3]
            ]
        )
        mm2.register(a1)
        b2 = Bias([2],[2])
        b2._b = np.zeros(2)
        b2.register(mm2)
        a2 = Sigmoid()
        a2.register(b2)
        loss = MSE([2])
        loss.register_ground_truth(_ground_truth)
        _ground_truth._forward_pass()
        loss.register_output(a2)
        return [_input, _ground_truth, mm1, b1, a1, mm2, b2, a2, loss]

    def test_forward_pass(self):
        # initialize
        _input, _, _, b1, a1, _, b2, a2, _ = self._prepare_test_case()

        # forward pass
        _input._forward_pass()

        # check result
        self.assertAlmostEqual(b1.output[0][0], -9.93)
        self.assertAlmostEqual(b1.output[0][1], 1.6)
        self.assertAlmostEqual(b1.output[0][2], 2.51)
        self.assertAlmostEqual(a1.output[0][0], 4.8689425323061825e-05)
        self.assertAlmostEqual(a1.output[0][1], 0.8320183851339245)
        self.assertAlmostEqual(a1.output[0][2], 0.9248398905178734)
        self.assertAlmostEqual(b2.output[0][0], 1.0170009700650962)
        self.assertAlmostEqual(b2.output[0][1], 2.9408503144424207)
        self.assertAlmostEqual(a2.output[0][0], 0.7343880132771364)
        self.assertAlmostEqual(a2.output[0][1], 0.949829262885629)
    
    def test_back_prop(self):
        # initialize
        _input, _, mm1, b1, _, mm2, b2, _, loss = self._prepare_test_case()
        _input._forward_pass()

        # back prop
        loss._back_prop()

        # check result
        self.assertAlmostEqual(b2._b[0], 0.10362174841852187)
        self.assertAlmostEqual(b2._b[1], -0.09052563259036336)
        self.assertAlmostEqual(mm2._w[0][0], 0.30000504528338146)
        self.assertAlmostEqual(mm2._w[1][0], 1.0862151997839324)
        self.assertAlmostEqual(mm2._w[2][0], 0.29583352646265637)
        self.assertAlmostEqual(mm2._w[0][1], -1.500004407641028)
        self.assertAlmostEqual(mm2._w[1][1], 0.12468100935893892)
        self.assertAlmostEqual(mm2._w[2][1], 2.9162782838660672)
        self.assertAlmostEqual(b1._b[0], 8.12465095e-06)
        self.assertAlmostEqual(b1._b[1], 1.19521274e-02)
        self.assertAlmostEqual(b1._b[2], -1.74370284e-02)
        self.assertAlmostEqual(mm1._w[0][0], 0.10000569)
        self.assertAlmostEqual(mm1._w[1][0], -4.99998375)
        self.assertAlmostEqual(mm1._w[0][1], 2.00836649)
        self.assertAlmostEqual(mm1._w[1][1], 0.12390425)
        self.assertAlmostEqual(mm1._w[0][2], -0.71220592)
        self.assertAlmostEqual(mm1._w[1][2], 1.46512594)


if __name__ == '__main__':
    unittest.main()
