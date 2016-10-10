import numpy as np
from pyfunt import Module


class DeepDreamLoss(Module):

    def __init__(self, strength=1e-5, max_grad=100.0):
        super(DeepDreamLoss, self).__init__()
        self.strength = strength
        self.max_grad = max_grad
        self.clipped = np.ndarray([])
        self.loss = 0

    def update_output(self, x):
        self.output = x
        return self.output

    def update_grad_input(self, x, grad_output):
        max_value = np.max(-self.max_grad, self.max_grad)
        min_value = np.min(-self.max_grad, self.max_grad)
        self.clipped = np.maximum(np.minimum(x, max_value), min_value)
        self.grad_input = grad_output.copy()
        self.grad_input -= self.strength * self.clipped
        return self.grad_input

    def reset(self):
        pass
