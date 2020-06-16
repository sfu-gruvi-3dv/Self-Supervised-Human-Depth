import numpy as np

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import checkpoint_utils


def preprocess_image(image):
    # Assuming input image is uint8
    if image == None:
        return None
    else:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. - 1.


def scan_checkpoint_for_vars(checkpoint_path, vars_to_check):
    check_var_list = checkpoint_utils.list_variables(checkpoint_path)
    check_var_list = [x[0] for x in check_var_list]
    check_var_set = set(check_var_list)
    vars_in_checkpoint = [x for x in vars_to_check if x.name[:x.name.index(":")] in check_var_set]
    vars_not_in_checkpoint = [x for x in vars_to_check if x.name[:x.name.index(":")] not in check_var_set]
    return vars_in_checkpoint, vars_not_in_checkpoint


def split_Rt(matrix34):
    if len(matrix34.shape) > 2:
        R = matrix34[:, :, :3, :3]
        t = matrix34[:, :, :3, 3]
    else:
        R = matrix34[:3, :3]
        t = matrix34[:3, 3]
    return R, t


def read_instrinsic_matrix():
    # intrinsic_mat = np.array((329.7, 0, 96.27,
    #                           0 , 329.7, 63.30,
    #                           0, 0, 1))
    # intrinsic = np.array((442.5352 / 2, 0, 256 / 2,
    #                       0, 442.5352 / 2, 256 / 2,
    #                       0, 0, 1))
    # intrinsic = np.array((222.2364, 0, 130.95,
    #                       0, 222.2364, 126.53,
    #                       0, 0, 1))
    intrinsic = np.array((366.4744, 0, 128.00,
                          0, 366.4744, 128.00,
                          0, 0, 1))
    intrinsic.resize((3, 3))
    # inv = np.linalg.inv(intrinsic)
    return intrinsic


def gaussian_pyramid(image, half_size=5, mean=0, std=2.5, num_scales=5):
    def gaussian_kernel(size: int, mean: float, std: float):
        """Makes 2D gaussian Kernel for convolution."""

        d = tf.distributions.Normal(float(mean), float(std))

        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    _, h, w, c = image.get_shape().as_list()

    scaled_imgs = []

    for i in range(num_scales):
        ratio = 2 ** (i)
        nh = int(h / ratio)
        nw = int(w / ratio)

        nstd = std

        # Make Gaussian Kernel with desired specs.
        gauss_kernel = gaussian_kernel(int(nstd*3)+1, mean, nstd)

        # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
        gauss_kernel = tf.tile(gauss_kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, c, 1])

        resized_image = tf.nn.depthwise_conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

        resized_image = tf.image.resize_nearest_neighbor(resized_image, [nh, nw])

        scaled_imgs.append(resized_image)

        image = resized_image

    return scaled_imgs


def gaussian_pyramid(image, mean=0, std=0.3, num_scales=5):
    def gaussian_kernel(size: int, mean: float, std: float):
        """Makes 2D gaussian Kernel for convolution."""

        d = tf.distributions.Normal(float(mean), float(std))

        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    _, h, w, c = image.get_shape().as_list()

    scaled_imgs = [image]

    for i in range(num_scales-1):
        ratio = 2 ** (i+1)
        nh = int(h / ratio)
        nw = int(w / ratio)

        nstd = std

        # Make Gaussian Kernel with desired specs.
        gauss_kernel = gaussian_kernel(int(nstd*3)+1, mean, nstd)

        # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
        gauss_kernel = tf.tile(gauss_kernel[:, :, tf.newaxis, tf.newaxis], [1, 1, c, 1])

        resized_image = tf.nn.depthwise_conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

        resized_image = tf.image.resize_bilinear(resized_image, [nh, nw])

        scaled_imgs.append(resized_image)

        image = resized_image

    return scaled_imgs


def low_pass_filter(image, mean=0, std=0.3, gamma=2.8):
    def gaussian_kernel(size: int, mean: float, std: float):
        """Makes 2D gaussian Kernel for convolution."""

        d = tf.distributions.Normal(float(mean), float(std))

        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    _, h, w, c = image.get_shape().as_list()

    nstd = std

    half_size = 5

    # Make Gaussian Kernel with desired specs.
    gauss_kernel = gaussian_kernel(half_size, mean, nstd)

    gauss_kernel = tf.reshape(gauss_kernel, [1,1,1,(half_size * 2 + 1)*(half_size * 2 + 1)])

    extracted_image = tf.image.extract_image_patches(image, ksizes=[1, half_size * 2 + 1, half_size * 2 + 1, 1], strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')

    intensity_weight=tf.math.exp(tf.math.square(image - extracted_image) * gamma**2)

    weight = gauss_kernel * intensity_weight

    resized_image = tf.reduce_sum(extracted_image * weight, axis=-1, keepdims=True) / tf.reduce_sum(weight + 0.00001, axis=-1, keepdims=True)

    low_pass_imgs = tf.stop_gradient(resized_image)

    return low_pass_imgs


def masked_huberloss(pred_value, gt_value, mask, delta=0.3):
    error = tf.abs(pred_value - gt_value)
    error = tf.boolean_mask(error, mask)
    delta = tf.constant(delta, name='max_grad')
    lin = delta * (error - 0.5 * delta)
    quad = 0.5 * error * error
    return tf.reduce_mean(tf.where(error < delta, quad, lin))



