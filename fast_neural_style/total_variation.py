from pyfunt import Module
import numpy as np


class TotalVariation(Module):

    def __init__(self, strength):
        super(TotalVariation, self).__init__()
        self.strength = strength

    def update_output(self, x):
        self.output = x
        return self.output

    def update_grad_input(self, x, grad_output):
        # -- TV loss backward pass inspired by kaishengtai/neuralart
        self.grad_input = np.zeros_like(x)
        N, C, H, W = x.shape
        self.x_diff = x[:, :, 1:-2, 1:-2]
        self.x_diff -= x[:, :, 1:-2, 2:-1]
        self.y_diff = x[:, :, 1:-2, 1:-2]
        self.y_diff -= x[:, :, 2:-1, 1:-2]
        self.grad_input[:, :, 1:-2, 1:-2] += self.x_diff + self.y_diff
        self.grad_input[:, :, 1:-2, 2:-1] -= self.x_diff
        self.grad_input[:, :, 2:-1, 1:-2] -= self.y_diff
