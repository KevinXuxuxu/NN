import numpy as np

from activation import Activation, Sigmoid

class Layer:

    def __init__(self, n: int, learning_rate: float, act: Activation = Sigmoid):
        self.n = n # dimension
        self.next, self.prev = None, None
        self._act = act # activation function
        self._rate = learning_rate

    def connect(self, prev_layer: 'Layer'):
        self.prev, prev_layer.next = prev_layer, self
        self._w = np.random.random((self.n, prev_layer.n)) # weights
        self._b = np.random.random((self.n, 1)) # bias

    def forward_pass(self):
        # a_L = act(w_L \dot a_(L-1) + b_L)
        if self.prev:
            # neuron values before activation
            self._z = np.matmul(self._w, self.prev.a) + self._b
            # neuron values
            self.a = self._act.eval(self._z)
        if self.next:
            self.next.forward_pass()

    def back_prop(self, a_grad: np.ndarray):
        # a_grad.shape == (n, p)
        if self.prev:
            # (n, p) = (n, p) * (n, p)
            b_grad = a_grad * self._act.grad(self._z) # * 1.
            # (n, m) = einsum('ik,jk', (n, p), (m, p))
            w_grad = np.einsum('ik,jk', b_grad, self.prev.a)
            # (m, p) = einsum('ij,ik', (n, m), (n, p))
            prev_a_grad = np.einsum('ij,ik', self._w, b_grad)
            self._b -= self._rate * np.sum(b_grad, axis=1)[:, np.newaxis]
            self._w -= self._rate * w_grad
            self.prev.back_prop(prev_a_grad)
