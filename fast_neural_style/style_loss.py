import pyfunt


class StyleLoss(pyfunt.Module):

    def __init__(self, strength=1.0, loss_type='L2', agg_type='gram'):
        super(StyleLoss, self).__init__()
        self.agg_type = agg_type
        self.strength = strength
        self.loss = 0.0
        if self.agg_type == 'gram':
            self.agg = pyfunt.GramMatrix()
        elif self.agg_type == 'mean':
            self.agg = pyfunt.Sequential()
            self.agg.add(pyfunt.Mean(3))
            self.agg.add(pyfunt.Mean(3))
        else:
            raise Exception('Unsupported aggregation type: ' + agg_type)
        self.mode = 'none'
        if loss_type == 'L2':
            self.crit = pyfunt.MSECriterion()
        elif loss_type == 'SmoothL1':
            self.crit = pyfunt.SmoothL1Criterion()
        else:
            raise Exception('invalid loss type "%s"' % loss_type)

    def update_output(self, x):
        self.agg_out = self.agg.forward(x)
        if self.mode == 'capture':
            self.target = self.agg_out.copy()
        elif self.mode == 'loss':
            target = self.target
            if self.agg_out.shape[0] > 1 and self.target.shape[0] == 1:
                # Handle minibatch inputs
                target = target.expand_as(self.agg_out)
            self.loss = self.strength * self.crit(self.agg_out, target)
            self._target = target
        self.output = x
        return self.output

    def update_grad_input(self, x, grad_output):
        if self.mode == 'capture' or self.mode == 'none':
            self.grad_input = grad_output
        elif self.mode == 'loss':
            self.crit.backward(self.agg_out, self.target)
            self.crit.grad_input *= self.strength
            self.agg.backward(x, self.crit.grad_input)
            self.grad_input += self.agg.grad_input * grad_output
        return self.grad_input

    def set_mode(self, mode):
        if mode != 'capture' and mode != 'loss' and mode != 'none':
            raise Exception('Invalid mode "%s"' % mode)

