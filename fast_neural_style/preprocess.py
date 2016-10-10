import pyfunt
import numpy as np


def check_input(img):
    if img.ndim != 4:
        raise Exception('img must be N x C x H x W')
    if img.shape[1] != 3:
        raise Exception('img must have three channels')

resnet_mean = np.array([0.485, 0.456, 0.406])
resnet_std = np.array([0.229, 0.224, 0.225])


'''
Preprocess an image before passing to a ResNet model. The preprocessing is easy:
we just need to subtract the mean and divide by the standard deviation. These
constants are taken from fb.resnet.torch:

https://github.com/facebook/fb.resnet.torch/blob/master/datasets/imagenet.lua

Input:
- img: Tensor of shape (N, C, H, W) giving a batch of images. Images are RGB
  in the range [0, 1].
'''


def resnet_preprocess(img):
    check_input(img)
    mean = img.new(resnet_mean).reshape(1, 3, 1, 1)
    std = img.new(resnet_mean).reshape(1, 3, 1, 1)
    return (img - mean) / std


def resnet_deprocess(img):
    check_input(img)
    mean = img.new(resnet_mean).reshape(1, 3, 1, 1)
    std = img.new(resnet_std).reshape(1, 3, 1, 1)
    return (img * std) + mean


vgg_mean = np.array([103.939, 116.779, 123.68])

# Preprocess an image before passing to a VGG model. We need to rescale from
# [0, 1] to [0, 255], convert from RGB to BGR, and subtract the mean.

# Input:
# - img: Tensor of shape (N, C, H, W) giving a batch of images. Images


def vgg_preprocess(img):
    check_input(img)
    return (img * 255) - vgg_mean.reshape(1, 3, 1, 1)


def vgg_deprocess(img):
    check_input(img)
    return ((img + vgg_mean.reshape(1, 3, 1, 1)) / 255)
