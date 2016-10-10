import pyfunt
from instance_normalization import InstanceNormalization
from shave_image import ShaveImage
from total_variation import TotalVariation


def build_conv_block(dim, padding_type, use_instance_norm):
    conv_block = pyfunt.Sequential()
    p = 0
    if padding_type == 'reflect':
        conv_block.add(pyfunt.SpatialReflectionPadding(1, 1, 1, 1))
    elif padding_type == 'replicate':
        conv_block.add(pyfunt.SpatialReplicationPadding(1, 1, 1, 1))
    elif padding_type == 'zero':
        p = 1
    conv_block.add(pyfunt.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
    if use_instance_norm == 1:
        conv_block.add(InstanceNormalization(dim))
    else:
        conv_block.add(pyfunt.SpatialBatchNormalization(dim))
    conv_block.add(pyfunt.ReLU(True))
    if padding_type == 'reflect':
        conv_block.add(pyfunt.SpatialReflectionPadding(1, 1, 1, 1))
    elif padding_type == 'replicate':
        conv_block.add(pyfunt.SpatialReplicationPadding(1, 1, 1, 1))
    conv_block.add(pyfunt.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
    if use_instance_norm == 1:
        conv_block.add(InstanceNormalization(dim))
    else:
        conv_block.add(pyfunt.SpatialBatchNormalization(dim))
    return conv_block


def build_res_block(dim, padding_type, use_instance_norm):
    conv_block = build_conv_block(dim, padding_type, use_instance_norm)
    res_block = pyfunt.Sequential()
    concat = pyfunt.ConcatTable()
    concat.add(conv_block)
    if padding_type == 'none' or padding_type == 'reflect-start':
        concat.add(ShaveImage(2))
    else:
        concat.add(pyfunt.Identity())
    res_block.add(concat).add(pyfunt.CAddTable())
    return res_block


def build_model(opt):
    arch = opt.arch.split(',')
    prev_dim = 3
    model = pyfunt.Sequential()

    for i, v in enumerate(arch):
        first_char = v[0]
        needs_relu = True
        needs_bn = True
        if v[0] == 'c':
            # Convolution
            f = int(v[1])  # filter size
            p = (f - 1) / 2  # padding
            s = int(v[3])  # stride
            next_dim = int(v[6])  # ?
            if opt.padding_type == 'reflect':
                model.add(pyfunt.SpatialReflectionPadding(p, p, p, p))
            elif opt.padding_type == 'replicate':
                model.add(pyfunt.SpatialReplicationPadding(p, p, p, p))
            p = 0
            layer = pyfunt.SpatialConvolution(
                prev_dim, next_dim, f, f, s, s, p, p)
        elif first_char == 'f':
            # Full Convolution
            f = int(v[1])  # filter size
            p = (f - 1) / 2  # padding
            s = int(v[3])  # stride
            a = s - 1  # adjustements
            next_dim = int(v[5])
            layer = pyfunt.SpatialFullConvolution(
                prev_dim, next_dim, f, f, s, s, p, p, a, a)
        elif first_char == 'd':
            # Downsampling (strided convolution)
            next_dim = int(v[1])
            layer = pyfunt.SpatialConvolution(prev_dim, next_dim, 3, 3, 2, 2, 1, 1)
        elif first_char == 'U':
            # Nearest-neighbor upsampling
            next_dim = prev_dim
            scale = int(v[1])
            layer = pyfunt.SpatialUpSamplingNearest(scale)
        elif first_char == 'u':
            # Learned upsampling (strided full-convolution)
            next_dim = int(v[1])
            layer = pyfunt.SpatialFullConvolution(prev_dim, next_dim, 3, 3, 2, 2, 1, 1, 1, 1)
        elif first_char == 'C':
            # Non-residual conv block
            next_dim = int(v[1])
            layer = build_conv_block(next_dim, opt.padding_type, opt.use_instance_norm)
            needs_bn = False
            needs_relu = True
        elif first_char == 'R':
            # Residual (non-bottleneck) block
            next_dim = int(v[1])
            layer = build_res_block(next_dim, opt.padding_type, opt.use_instance_norm)
            needs_bn = False
            needs_relu = True
        model.add(layer)
        if i == len(arch)-1:
            needs_relu = False
            needs_bn = False
        if needs_bn:
            if opt.use_instance_norm == 1:
                model.add(InstanceNormalization(next_dim))
            else:
                model.add(pyfunt.SpatialBatchNormalization(next_dim))
        if needs_relu:
            model.add(pyfunt.ReLU(True))
        prev_dim = next_dim
    model.add(pyfunt.Tanh())
    model.add(pyfunt.MulConstant(opt.tanh_constant))
    model.add(TotalVariation(opt.tv_strength))
    return model
