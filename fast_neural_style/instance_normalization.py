import numpy as np
import pyfunt


class InstanceNormalization(pyfunt.Module):

    #   Implements instance normalization as described in the paper

    #   Instance Normalization: The Missing Ingredient for Fast Stylization
    #   Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
    #   https://arxiv.org/abs/1607.08022

    #   This implementation is based on
    #   https://github.com/DmitryUlyanov/texture_nets

    def __init__(self, n_output, eps=1e-5):
        super(InstanceNormalization, self).__init__()
        self.n_output = n_output
        self.eps = eps
        self.prev_N = -1
        self.weight = np.random.uniform(size=n_output)
        self.bias = np.zeros(n_output)
        self.grad_weight = np.zeros(n_output)
        self.grad_weight = np.zeros(n_output)
        self.dn = False

    def update_output(self, x):
        N, C, H, W = x.shape
        if C != self.n_output:
            raise Exception('C != self.n_output')
        if N != self.prev_N or self.bn:
            self.bn = pyfunt.SpatialBatchNormalization(N * C, self.eps)
            self.prev_N = N

        input_view = x.reshape(1, N * C, H, W)

        self.bn.training()
        self.output = self.bn.forward(input_view).reshape(x.shape)
        return self.output

    def update_grad_input(self, x, grad_output):
        N, C, H, W = x.shape
        if self.bn is None:
            raise
        input_view = x.reshape(1, N*C, H, W)
        grad_output_view = grad_output.reshape(1, N*C, H, W)

        self.bn.grad_weight *= 0.0
        self.bn.grad_bias *= 0.0

        self.bn.training()
        self.grad_input = self.bn.backward(input_view, grad_output_view).reshape(x.shape)

        self.grad_weight += np.sum(self.bn.grad_bias.reshape(N, C), axis=0)
        self.grad_bias += np.sum(self.bn.grad_bias.reshape(N, C), axis=0)

        return self.grad_input

    def clear_state(self):
        self.output = np.zeros_like(self.output)
        self.grad_input = np.zeros_like(self.grad_input)
        self.bn.clear_state()
