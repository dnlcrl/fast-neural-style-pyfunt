import numpy as np
from content_loss import ContentLoss
import layer_utils
from style_loss import StyleLoss
from deep_dream_loss import DeepDreamLoss
from pyfunt import Criterion

'''
NOT TESTED!!!
'''


class PerceptualCriterion(Criterion):
    '''Input: args is a table with the following keys:
    cnn: A network giving the base CNN.
    content_layers: An array of layer strings
    content_weights: A list of the same length as content_layers
    style_layers: An array of layers strings
    style_weights: A list of the same length as style_layers
    agg_type: What type of spatial aggregaton to use for style loss;
    "mean" or "gram"
    deepdream_layers: Array of layer strings
    deepdream_weights: List of the same length as deepdream_layers
    loss_type: Either "L2", or "SmoothL1"'''

    def __init__(self, args):
        super(PerceptualCriterion, self).__init__()
        args.content_layers = args.content_layers or []
        args.style_layers = args.style_layers or []
        args.deepdream_layers = args.deepdream_layers or []
        self.net = args.cnn
        self.net.evaluate()
        self.content_loss_layers = []
        self.style_loss_layers = []
        self.deepdream_loss_layers = []
        # Set up content loss layers
        for i, layer_string in enumerate(args.content_layers):
            weight = args.content_weights[i]
            content_loss_layer = ContentLoss(weight, args.loss_type)
            layer_utils.insert_after(self.net, layer_string, content_loss_layer)
            self.content_loss_layers.append(content_loss_layer)

        # Set up style loss layers
        for i, layer_string in enumerate(args.style_layers):
            weight = args.style_weights[i]
            style_loss_layer = StyleLoss(weight, args.loss_type)
            layer_utils.insert_after(self.net, layer_string, style_loss_layer)
            self.style_loss_layers.append(style_loss_layer)

        # Set up DeepDream layers
        for i, layer_string in enumerate(args.deepdream_layers):
            weight = args.deepdream_weights[i]
            deepdream_loss_layer = DeepDreamLoss(weight)
            layer_utils.insert_after(self.net, layer_string, deepdream_loss_layer)
            self.deepdream_loss_layers.append(deepdream_loss_layer)

        layer_utils.trim_network(self.net)

    def set_style_target(self, target):
        # target: Tensor of shape (1, 3, H, W) giving pixels for style target image
        for content_loss_layer in self.content_loss_layers:
            content_loss_layer.set_mode('none')
        for style_loss_layer in self.style_loss_layers:
            style_loss_layer.set_mode('capture')
        self.net.forward(target)

    def set_content_target(self, target):
        # target: Tensor of shape (N, 3, H, W) giving pixels for content target images
        for style_loss_layer in self.style_loss_layers:
            style_loss_layer.set_mode('none')
        for content_loss_layer in self.content_loss_layers:
            content_loss_layer.set_mode('capture')
        self.net.forward(target)

    def set_style_weight(self, weight):
        for style_loss_layer in self.style_loss_layers:
            style_loss_layer.strength = weight

    def set_content_weight(self, weight):
        for content_loss_layer in self.content_loss_layers:
            content_loss_layer.strength = weight

    def update_output(self, x, target):
        # Inputs:
        # - input: Tensor of shape (N, 3, H, W) giving pixels for generated images
        # - target: Table with the following keys:
        #   - content_target: Tensor of shape (N, 3, H, W)
        #   - style_target: Tensor of shape (1, 3, H, W)
        if target.content_target:
            self.set_content_target(target.content_target)
        if target.style_target:
            self.set_style_target(target.style_target)
        # Make sure to set all content and style loss layers to loss mode before
        # running the image forward.
        for content_loss_layer in self.content_loss_layers:
            content_loss_layer.set_mode('loss')
        for style_loss_layer in self.style_loss_layers:
            style_loss_layer.set_mode('loss')
        output = self.net.forward(x)
        # Set up a tensor of zeros to pass as gradient to net in backward pass
        self.grad_net_output = np.zeros_like(output)
        # Go through and add up losses
        self.total_style_loss = 0
        self.total_content_loss = 0
        self.content_losses = []
        self.style_losses = []
        for content_loss_layer in self.content_loss_layers:
            self.total_content_loss = self.total_content_loss + content_loss_layer.loss
            self.content_losses.append(content_loss_layer.loss)
        for style_loss_layer in self.style_loss_layers:
            self.total_style_loss = self.total_style_loss + style_loss_layer.loss
            self.style_losses.append(style_loss_layer.loss)
        self.output = self.total_style_loss + self.total_content_loss
        return self.output

    def update_grad_input(self, x, target):
        self.gradInput = self.net.update_grad_input(x, self.grad_net_output)
        return self.grad_input

