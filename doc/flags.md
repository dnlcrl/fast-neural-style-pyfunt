## fast_neural_style.py

The script `fast_neural_style.py` runs a trained model on new images. It has
the following flags:

**Model options**:
- `-model`: Path to a `.t7` model file from `train.lua`.
- `-image_size`: Before being input to the network, images are resized so
  their longer side is this many pixels long. If 0 then use the original size
  of the image.
- `-median_filter`: If nonzero, use a
  [median filter](https://en.wikipedia.org/wiki/Median_filter) of this kernel
  size as a post-processing step. Default is 3.

**Input / Output**:
- `-input_image`: Path to a single image on which to run the model.
- `-input_dir`: Path to a directory of image files; the model will be run
  on all images in the directory.
- `-output_image`: When using `-input_image` to specify input, the
  path to which the stylized image will be written.
- `-output_dir`: When using `-input_dir` to specify input, this gives a path
  to a directory where stylized images will be written. Each output image
  will have the same filename as its corresponding input image.
