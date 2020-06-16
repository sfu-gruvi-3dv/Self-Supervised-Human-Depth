#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  23/10/18 7:23 PM
#  feitongt
#  photo_loss.py

import tensorflow as tf
import tensorflow.contrib.slim as slim

alpha_recon_image = 1.0

def RGB_to_grey(x):
    # color_filter = tf.constant((0.3, 0.59, 0.11))
    # color_filter = tf.reshape(color_filter, [1, 1, 3, 1])
    #
    # x = tf.nn.conv2d(x, color_filter, strides=[1, 1, 1, 1], padding='SAME')
    return x


def SSIM(x, y, size=3):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, size, 1, 'SAME')
    mu_y = slim.avg_pool2d(y, size, 1, 'SAME')

    sigma_x = slim.avg_pool2d(x ** 2, size, 1, 'SAME') - mu_x ** 2
    sigma_y = slim.avg_pool2d(y ** 2, size, 1, 'SAME') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y, size, 1, 'SAME') - mu_x * mu_y
    #
    # SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    # SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM_n = (2 * sigma_xy + C2)
    SSIM_d =  (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    SSIM_Loss = tf.clip_by_value((1 - SSIM) / 2, 1e-8, 1.0)
    SSIM_Loss = tf.reduce_mean(SSIM_Loss, axis=-1, keep_dims=True)

    x_grey = RGB_to_grey(x)
    y_grey = RGB_to_grey(y)
    mu_x = slim.avg_pool2d(x_grey, 4, 1, 'SAME')
    mu_y = slim.avg_pool2d(y_grey, 4, 1, 'SAME')

    intensity_diff = tf.reduce_mean(tf.abs(mu_x - mu_y), axis=-1, keep_dims=True)
    SSIM_Loss = tf.where(intensity_diff < 0.4, SSIM_Loss, tf.zeros_like(SSIM_Loss))
    SSIM_Loss = tf.where(SSIM_Loss<0.85, SSIM_Loss, tf.zeros_like(SSIM_Loss))

    return SSIM_Loss


def image_similarity(x, y):

    x_grey = RGB_to_grey(x)
    y_grey = RGB_to_grey(y)

    mu_x = slim.avg_pool2d(x_grey, 4, 1, 'SAME')
    mu_y = slim.avg_pool2d(y_grey, 4, 1, 'SAME')

    intensity_diff = tf.abs(mu_x - mu_y)
    mask = intensity_diff < 0.3
    photo_loss = tf.reduce_mean(tf.abs(x - y), axis=-1, keep_dims=True)
    photo_loss = tf.where(mask, photo_loss, tf.zeros_like(photo_loss))

    return photo_loss


def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
      batch: batch number
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    Returns:
      x,y grid coordinates [height, width, 2 (3 if homogeneous)]
    """
    x_t, y_t = tf.meshgrid(tf.linspace(0.0, width - 1.0, width),
                           tf.linspace(0.0, height - 1.0, height))
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=-1)
        coords = tf.expand_dims(coords, axis=0)
        coords = tf.tile(coords, (batch, 1, 1, 1))
    else:
        coords = tf.stack([x_t, y_t], axis=-1)
        coords = tf.expand_dims(coords, axis=0)
        coords = tf.tile(coords, (batch, 1, 1, 1))
    return coords


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
      depth: [batch, height, width]
      pixel_coords: homogeneous pixel coordinates [batch, height, width, 3]
      intrinsics: camera intrinsics [height, width, 3, 3]
      is_homogeneous: return in homogeneous coordinates
    Returns:
      Coords in the camera frame [height, width, 3 (4 if homogeneous)]
    """
    batch, height, width, _ = depth.get_shape().as_list()
    intrinsics_inv = tf.matrix_inverse(intrinsics)
    pixel_coords = tf.matmul(intrinsics_inv, pixel_coords[..., tf.newaxis])[..., 0]
    cam_coords = tf.reshape(pixel_coords, [-1, height, width, 3]) * depth

    if is_homogeneous:
        ones = tf.ones_like(cam_coords)
        cam_coords = tf.concat([cam_coords, ones], axis=-1)
    return cam_coords


def cam2pixel(cam_coords, intrinsics):
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
      cam_coords: [height, width]
      proj: [height, width, 3, 3]
    Returns:
      Pixel coordinates projected from the camera frame [height, width, 3]
    """
    _, height, width, _ = cam_coords.get_shape().as_list()
    cam_coords = tf.matmul(intrinsics, cam_coords[..., tf.newaxis])[..., 0]
    pixel_coords = cam_coords[..., 0:2] / tf.clip_by_value(cam_coords[..., 2:3], 1e-6, 1e6)

    return pixel_coords


def shape_deformation(src_coords, R_mat, t_mat):
    src_coords = tf.expand_dims(src_coords, axis=-1)
    dfm_coords = tf.squeeze(tf.matmul(R_mat, src_coords), -1) + t_mat
    return dfm_coords


def compute_nonrigid_flow(depth, R_mat, t_mat, intrinsics):
    batch_ref_num, height, width, _ = depth.get_shape().as_list()

    # Construct pixel grid coordinates
    pixel_coords = meshgrid(batch_ref_num, height, width, True)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics, False)
    dfm_cam_coords = shape_deformation(cam_coords, R_mat, t_mat)
    dfm_pixel_coords = cam2pixel(dfm_cam_coords, intrinsics)


    return dfm_pixel_coords


def bilinear_sampler(imgs, coords):
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [height_t, width_t, channels]
    """
    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)

        img_size_int = imgs.get_shape().as_list()
        coord_size_int = coords.get_shape().as_list()
        batch_size = img_size_int[0]
        img_height = img_size_int[1]
        img_width = img_size_int[2]
        coord_height = coord_size_int[1]
        coord_width = coord_size_int[2]

        coords_y = tf.cast(coords_y, 'float32')
        coords_x = tf.cast(coords_x, 'float32')

        coords_y = coords_y / coord_height * img_height
        coords_x = coords_x / coord_width * img_width

        # ### implementation
        y_max = img_height - 1
        x_max = img_width - 1

        coords_y = tf.clip_by_value(coords_y, 0, y_max)
        coords_x = tf.clip_by_value(coords_x, 0, x_max)
        coords = tf.concat([coords_x, coords_y], axis=-1)
        output = tf.contrib.resampler.resampler(imgs, coords)

        # ### original version
        # y_max = img_height - 1
        # x_max = img_width - 1
        #
        # coords_y = tf.clip_by_value(coords_y, 0, y_max)
        # coords_x = tf.clip_by_value(coords_x, 0, x_max)
        #
        # x0 = tf.floor(coords_x)
        # x1 = x0 + 1
        # y0 = tf.floor(coords_y)
        # y1 = y0 + 1
        #
        # x0_safe = tf.clip_by_value(x0, 0, x_max)
        # y0_safe = tf.clip_by_value(y0, 0, y_max)
        # x1_safe = tf.clip_by_value(x1, 0, x_max)
        # y1_safe = tf.clip_by_value(y1, 0, y_max)
        #
        # wt_x0 = x1_safe - coords_x
        # wt_x1 = coords_x - x0_safe
        # wt_y0 = y1_safe - coords_y
        # wt_y1 = coords_y - y0_safe
        #
        # batch_indices = tf.tile(tf.reshape(tf.range(batch_size, dtype=tf.float32), (-1, 1, 1, 1)),
        #                         (1, coord_height, coord_width, 1))
        #
        # idx00 = tf.concat([batch_indices, y0_safe, x0_safe], axis=-1)
        # idx01 = tf.concat([batch_indices, y1_safe, x0_safe], axis=-1)
        # idx10 = tf.concat([batch_indices, y0_safe, x1_safe], axis=-1)
        # idx11 = tf.concat([batch_indices, y1_safe, x1_safe], axis=-1)
        #
        # im00 = tf.gather_nd(imgs, tf.cast(idx00, 'int32'))
        # im01 = tf.gather_nd(imgs, tf.cast(idx01, 'int32'))
        # im10 = tf.gather_nd(imgs, tf.cast(idx10, 'int32'))
        # im11 = tf.gather_nd(imgs, tf.cast(idx11, 'int32'))
        #
        # w00 = wt_x0 * wt_y0
        # w01 = wt_x0 * wt_y1
        # w10 = wt_x1 * wt_y0
        # w11 = wt_x1 * wt_y1
        #
        # output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11

    return output



def flow_warp(src_img, tgt_img, src_depth, R_mat, T_mat, intrinsics):
    height, width, _ = src_depth.get_shape().as_list()
    dfm_pixel_coords = compute_nonrigid_flow(src_depth, R_mat, T_mat, intrinsics)
    warpped_sampled_img = bilinear_sampler(tgt_img, dfm_pixel_coords[..., :2])
    pixel_coords = meshgrid(height, width, True)
    src_sampled_img = bilinear_sampler(src_img, pixel_coords[..., :2])

    return src_sampled_img, warpped_sampled_img, dfm_pixel_coords



def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy


def compute_smooth_loss(disp, img, mask_float):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

    # disp_gradients_x = tf.where(disp_gradients_x<threshold, disp_gradients_x, tf.zeros_like(disp_gradients_x))
    # disp_gradients_y = tf.where(disp_gradients_y<threshold, disp_gradients_y, tf.zeros_like(disp_gradients_y))

    smoothness_x = disp_gradients_x  #* weights_x
    smoothness_y = disp_gradients_y  #* weights_y

    return (tf.reduce_mean(tf.math.square(smoothness_x)) + tf.reduce_mean(tf.math.square(smoothness_y)))# / tf.reduce_sum(mask_float)
