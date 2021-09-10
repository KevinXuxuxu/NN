import numpy as np

from src.activation import Activation, Sigmoid

class Layer:

    def __init__(self, n: int, learning_rate: float, act: Activation = Sigmoid):
        self.n = n # dimension
        self.next, self.prev = None, None
        self._act = act # activation function
        self._rate = learning_rate

    def connect(self, prev_layer: 'Layer'):
        self.prev, prev_layer.next = prev_layer, self
        # weights initialized to -0.1 to 0.1
        self._w = np.random.random((prev_layer.n, self.n)) / 5. - .1
        # bias initialized to 0.1
        self._b = np.ones(self.n) * 0.1 # bias

    def forward_pass(self):
        # a_L = act(a_(L-1) \dot w_L + b_L)
        if self.prev:
            # neuron values before activation
            self._z = np.matmul(self.prev.a, self._w) + self._b
            # neuron values
            self.a = self._act.eval(self._z)
        if self.next:
            self.next.forward_pass()

    def back_prop(self, a_grad: np.ndarray):
        # a_grad.shape == (p, n)
        if self.prev:
            # (p, n) = (p, n) * (p, n) broadcast
            b_grad = a_grad * self._act.grad(self._z) # * 1.
            # (m, n) = einsum('ki,kj', (p, m), (p, n))
            w_grad = np.einsum('ki,kj', self.prev.a, b_grad)
            # (p, m) = einsum('ik,jk', (p, n), (m, n))
            prev_a_grad = np.einsum('ik,jk', b_grad, self._w)
            self._b -= self._rate * np.sum(b_grad, axis=0)
            self._w -= self._rate * w_grad
            self.prev.back_prop(prev_a_grad)
