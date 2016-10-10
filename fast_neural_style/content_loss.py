from module import Module
from pyfunt import (MSECriterion, SmoothL1Criterion)

# -- --[[
# -- Module to compute content loss in-place.

# -- The module can be in one of three modes: "none", "capture", or "loss", which
# -- behave as follows:
# -- - "none": This module does nothing; it is basically nn.Identity().
# -- - "capture": On the forward pass, inputs are captured as targets; otherwise it
# --   is the same as an nn.Identity().
# -- - "loss": On the forward pass, compute the distance between input and
# --   self.target, store the result in self.loss, and return input. On the backward
# --   pass, add compute the gradient of self.loss with respect to the inputs, and
# --   add this value to the upstream gradOutput to produce gradInput.
# -- --]]


class ContentLoss(Module):

    def __init__(self, strength=1.0, loss_type='L2'):
        super(ContentLoss, self).__init__()
        self.strength = strength
        self.loss = 0
        self.mode = 'none'
        if loss_type == 'L2':
            self.crit = MSECriterion()
        elif loss_type == 'SmoothL1':
            self.crit = SmoothL1Criterion()
        else:
            raise Exception('Invalid loss_type "%s"' % loss_type)

    def update_output(self, x):
        if self.mode == 'capture':
            self.target = x.copy()
        elif self.mode == 'loss':
            self.loss = self.strength * self.crit.forward(x, self.target)
        self.output = x
        return self.output

    def update_grad_input(self, x, grad_output):
        if self.mode == 'capture' or self.mode == 'none':
            self.grad_input = grad_output
        elif self.mode == 'loss':
            self.grad_input = self.crit.backward(x, self.target)
            self.grad_input *= self.strength
            self.grad_input += grad_output
        return self.grad_input

    def set_mode(self, mode):
        if mode not in ('capture', 'loss', 'none'):
            raise Exception('Invalid mode "%s"' % mode)
        self.mode = mode

    def reset(self):
        pass

