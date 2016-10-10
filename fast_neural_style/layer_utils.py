import pyfunt
import numpy as np

'''
Utility functions for getting and inserting layers into models composed of
hierarchies of nn Modules and nn Containers. In such a model, we can uniquely
address each module with a unique "layer string", which is a series of integers
separated by dashes. This is easiest to understand with an example: consider
the following network; we have labeled each module with its layer string:

nn.Sequential {
  (1) nn.SpatialConvolution
  (2) nn.Sequential {
    (2-1) nn.SpatialConvolution
    (2-2) nn.SpatialConvolution
  }
  (3) nn.Sequential {
    (3-1) nn.SpatialConvolution
    (3-2) nn.Sequential {
      (3-2-1) nn.SpatialConvolution
      (3-2-2) nn.SpatialConvolution
      (3-2-3) nn.SpatialConvolution
    }
    (3-3) nn.SpatialConvolution
  }
  (4) nn.View
  (5) nn.Linear
}

Any layers that that have the instance variable _ignore set to true are ignored
when computing layer strings for layers. This way, we can insert new layers into
a network without changing the layer strings of existing layers.
'''


def layer_string_to_nums(layer_string):
    # Convert a layer string to an array of integers.

    # For example layer_string_to_nums("1-23-4") = {1, 23, 4}.
    nums = []
    for s in layer_string.split('-'):
        nums.append(int(s))
    return nums


def compare_layer_strings(s1, s2):
    # Comparison function for layer strings that is compatible with table.sort.
    # In this comparison scheme, 2-3 comes AFTER 2-3-X for all X.

    # Input:
    # - s1, s2: Two layer strings.

    # Output:
    # - true if s1 should come before s2 in sorted order; false otherwise.
    left = layer_string_to_nums(s1)
    right = layer_string_to_nums(s2)
    out = None
    for i in range(np.max(len(left), len(right))):
        if left[i] < right[i]:
            out = True
        elif left[i] > right[i]:
            out = False
        if out is not None:
            break
    return out or len(left) > len(right)


def get_layer(net, layer_string):
    # Get a layer from the network net using a layer string.
    nums = layer_string_to_nums(layer_string)
    layer = net
    for num in nums:
        count = 0
        for j in range(len(layer)):
            if not layer.get(j)._ignore:
                count = count + 1
            if count == num:
                layer = layer.get(j)
    return layer


def insert_after(net, layer_string, new_layer):
    # Insert a new layer immediately after the layer specified by a layer string.
    # Any layers inserted this way are flagged with a special variable
    new_layer._ignore = True
    nums = layer_string_to_nums(layer_string)
    container = net
    for i in range(len(nums)):
        count = 0
        for j in range(len(container)):
            if not container.get(j)._ignore:
                count += 1
            if count == nums[i]:
                if i < range(len(nums)):
                    container = container.get(j)
                    break
                elif i == len(nums) - 1:
                    container.insert(new_layer, j+1)


def trim_network(net):
    # Remove the layers of the network that occur after the last _ignore
    def contains_ignore(layer):
        if type(layer) is pyfunt.Criterion:
            found = False
            for i in range(len(layer)):
                found = found or contains_ignore(layer.get[i])
            return found
        else:
            return layer._ignore is True

    last_layer = 0
    for i in range(len(net)):
        if contains_ignore(net.get[i]):
            last_layer = i
    num_to_remove = len(net) - 1 - last_layer
    for i in range(num_to_remove):
        net.remove()
    return net
