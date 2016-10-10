from pyfunt import Module
import numpy as np


class GramMatrix(Module):

    # Input:
    # - features: A tensor of shape (N, C, H, W) or (C, H, W) giving features for
    #   either a single image or a minibatch of images.

    # Output:
    # - gram: A tensor of shape (N, C, C) or (C, C) giving Gram matrix for input.
    # --]]
    def __init__(self, normalize=True):
        super(GramMatrix, self).__init__()
        self.normalize = normalize
        self.buffer = np.ndarray([])

    def update_output(self, x):
        if x.dims == 3:
            C, H, W = x.shape
            x_flat = x.view(C, H * W)
            self.output = np.dot(x_flat, x_flat.T)
        elif x.dims == 4:
            N, C, H, W = x.shape
            x_flat = x.view(N, C, H*W)
            self.output = np.tensordot(x_flat, x_flat.transpose(2, 3), 2)
        if self.normalize:
            self.output /= C * H * W
        return self.output

    def update_grad_input(self, x, grad_output):
        self.grad_input = np.zeros_like(x)
        if x.dims == 3:
            C, H, W = x.shape
            x_flat = x.view(C, H * W)
            self.buffer = np.dot(grad_output, x_flat)
            self.buffer += np.dot(grad_output.T, x_flat)
            self.grad_input = self.buffer.view(C, H, W)
        if x.dims == 4:
            N, C, H, W = x.shape
            x_flat = x.view(N, C, H * W)
            self.buffer = np.tensordot(grad_output, x_flat, 2)
            self.buffer += np.tensordot(grad_output.transpose(2, 3), x_flat, 2)
            self.grad_input = self.buffer.view(N, C, H, W)
        if self.normalize:
            self.buffer /= C * H * W
        return self.grad_input

    def reset(self):
        pass
