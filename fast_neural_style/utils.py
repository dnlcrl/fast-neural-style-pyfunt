import pyfunt


def parse_num_list(s):
    # Parse a string of comma-separated numbers
    # For example convert "1.0,3.14" to {1.0, 3.14}
    nums = []
    for ss in s.split(','):
        nums.append(int(ss))
    return nums


def layer_utils():
    # Parse a layer string and associated weights string.
    # The layers string is a string of comma-separated layer strings, and the
    # weight string contains comma-separated numbers. If the weights string
    # contains only a single number it is duplicated to be the same length as the
    # layers
    layers = layers_string.split(',')
    weights = parse_num_list(weights_string)
    if len(weights) == 1 and len(layers) > 1:
        # Duplicate the same weight for all layers
        w = weights[0]
        weights = []
        for i in range(len(layers)):
            weights.append(w)
    elif len(weights) != len(layers):
        msg = 'size mismatch between layers "%s" and weights "%s"'
        raise Exception(msg % (layers_string, weights_string))
    return layers, weights


def clear_gradients(m):
    if type(m) == pyfunt.Container:
        m.apply_to_modules(m.clear_gradients)
    if m.weight and m.grad_weight:
        m.grad_weight = m.grad_weight.new()
    if m.bias and m.grad_bias:
        m.grad_bias = m.grad_bias.new()


def restore_gradsients(m):
    if type(m) == pyfunt.Container:
        m.apply_to_modules(m.restore_gradients)
    if m.weight and m.grad_weight:
        m.grad_weight = np.zeros_like(m.grad_weight)
    if m.bias and m.grad_bias:
        m.grad_bias = np.zeros_like(m.grad_bias)


IMAGE_EXTS = ['jpg', 'jpeg', 'png', 'ppm', 'pgm']
def is_image_file(filename):
    # Hidden file are not images
    if filename[0] == '.':
        return False
    # Check against a list of known image extensions
    ext = filename.lower(paths.extname(filename))
    if ext in IMAGE_EXTS:
        return True
    return False


def median_filter(img, r):
    u = img.unfold(2, r, 1).contigous()
    u = u.unfold(3, r, 1).contigous()
    _, HH, WW = u.shape
    return u.view(3, HH, WW, r * r).median()
