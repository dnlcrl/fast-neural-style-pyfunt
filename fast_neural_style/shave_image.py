from pyfunt import Module
import numpy as np


class ShaveImage(Module):

    def __init__(self, shape):
        super(ShaveImage, self).__init__()
        self.shape = shape

    def update_output(self, x):
        s = self.shape
        N, C, H, W = x.shape
        self.output = x[:, :, s:H-s, s:W-s]
        return self.output

    def update_grad_input(self, x, grad_output):
        N, C, H, W = x.shape
        s = self.size
        self.grad_input = np.zeros_like(x)
        self.grad_input[:, :, s:H-s, s:W-s] = grad_output.copy()
        return self.grad_input
