import numpy as np
import unittest

from src.operator.operator import Operator
from src.operator.common import Bias, MatMul
from src.operator.activation import Sigmoid


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

    def test_forward_pass(self):
        # initialize
        # _input - MatMul1 - Bias1 - Sigmoid1 - MatMul2 - Bias2 - Sigmoid2
        _input = Operator([], [2])
        _input.output = np.array([[.7, 2.]])
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
    
    # TODO: add test_back_prop after cost operators are implemented


if __name__ == '__main__':
    unittest.main()
