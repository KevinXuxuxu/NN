import unittest

from src.operator import Operator


class Plus(Operator):

    def __init__(self, p: int):
        super().__init__([0], [0])
        self.p = p
        self.output = None

    def _forward_pass(self):
        self.output = sum(pred.output for pred in self.preds) + self.p
        super()._forward_pass()


class OperatorTest(unittest.TestCase):

    def test_forward_pass(self):
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


if __name__ == '__main__':
    unittest.main()
