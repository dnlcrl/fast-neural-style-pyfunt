import argparse
import pyfunt
from pyfunt.utils import (
    load_t7model, load_t7checkpoint, load_parser_init, load_parser_vals)
from fast_neural_style import (InstanceNormalization, ShaveImage, TotalVariation)
from fast_neural_style.preprocess import (resnet_preprocess, resnet_deprocess,
                                          vgg_preprocess, vgg_deprocess)
import numpy as np
from scipy.ndimage import imread
from scipy.misc import imresize
np.random.seed(0)
np.seterr(all='raise')
gb4 = 4*1024

import resource
resource.setrlimit(resource.RLIMIT_DATA, (gb4, gb4))

import gc


MODEL_PATH = 'models/instance_norm/candy.t7'
IMG_SIZE = 768
MEDIAN_FILTER = 3
TIMING = 0
INPUT_IMAGE = 'images/content/chicago.jpg'
OUTPUT_IMAGE = 'out.png'
INPUT_DIR = ''
OUTPUT_DIR = ''

opt = argparse.Namespace()


def parse_args():
    """
    Parse the options for running the
    """
    desc = ''
    parser = argparse.ArgumentParser(
        description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--model',
        metavar='PATH',
        default=MODEL_PATH,
        type=str,
        help='model path')
    add('--image_size',
        metavar='INT',
        default=IMG_SIZE,
        type=int,
        help='sife of the image')
    add('--median_filter',
        metavar='INT',
        default=MEDIAN_FILTER,
        type=int,
        help='median_filter')
    add('--timing',
        metavar='INT',
        default=TIMING,
        type=int,
        help='stiming')
    add('--input_image',
        metavar='PATH',
        default=INPUT_IMAGE,
        type=str,
        help='model path')
    add('--output_image',
        metavar='PATH',
        default=OUTPUT_IMAGE,
        type=str,
        help='model path')
    add('--input_dir',
        metavar='DIRECTORY',
        default=INPUT_DIR,
        type=str,
        help='model path')
    add('--output_dir',
        metavar='DIRECTORY',
        default=OUTPUT_DIR,
        type=str,
        help='model path')

    parser.parse_args(namespace=opt)


def instance_normalization_init(m):
    return m['nOutput'], m['eps']


def shave_image_init(m):
    return (m['size'],)


def total_variation_init(m):
    return (m['strength'],)

cload_parser_init = {
    'InstanceNormalization': instance_normalization_init,
    'ShaveImage': shave_image_init,
    'TotalVariation': total_variation_init,
}

load_parser_init.update(cload_parser_init)


def custom_layers(): return
custom_layers.InstanceNormalization = InstanceNormalization
custom_layers.ShaveImage = ShaveImage
custom_layers.TotalVariation = TotalVariation


def resnet(): return
resnet.preprocess = resnet_preprocess
resnet.deprocess = resnet_deprocess


def vgg(): return
vgg.preprocess = vgg_preprocess
vgg.deprocess = vgg_deprocess

methods = {'resnet': resnet, 'vgg': vgg}


def main():
    parse_args()
    if (opt.input_image == '') and (opt.input_dir == ''):
        raise Exception('Must give exactly one of -input_image or -input_dir')
    checkpoint = load_t7checkpoint(opt.model, custom_layers=custom_layers)
    model = checkpoint.model
    model.evaluate()
    gc.collect()
    preprocess_method = checkpoint.opt.preprocessing or 'vgg'
    preprocess = methods[preprocess_method]

    def run_image(in_path, out_path):
        img = imread(in_path)
        if opt.image_size > 0:
            img = imresize(img, np.float(opt.image_size)/img.shape[0])
        img = img.transpose(2, 0, 1).astype(np.float64)
        _, H, W = img.shape
        img = img.reshape(1, 3, H, W)
        img_pre = preprocess.preprocess(img)
        import pdb; pdb.set_trace()
        img_out = model.forward(img_pre)
        import pdb; pdb.set_trace()
        img_out = preprocess.deprocess(img_out)[0]
        if opt.median_filter > 0:
            img_out = utils.median_filter(img_out, opt.median_filter)
        print('Writing output image to ' + out_path)
        out_dir = paths.dirname(out_path)
        if not path.isdir(out_dir):
            paths.mkdir(outdir)
        image.save(out_path, img_out)

    if opt.input_dir != '':
        if opt.output_dir == '':
            raise Exception('Must give -output_dir with -input_dir')
        for fn in paths.files(opt.input_dir):
            if utils.is_image_file(fn):
                in_path = paths.concat(opt.input_dir, fn)
                out_path = paths.concat(opt.output_dir, fn)
                run_image(in_path, out_path)

    elif opt.input_image != '':
        if opt.output_image == '':
            raise Exception('Must give -output_image with -input_image')
        run_image(opt.input_image, opt.output_image)


if __name__ == '__main__':
    main()
