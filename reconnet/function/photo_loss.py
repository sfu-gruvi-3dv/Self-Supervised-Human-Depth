#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  23/10/18 7:23 PM
#  feitongt
#  photo_loss.py

import tensorflow as tf
import tensorflow.contrib.slim as slim

alpha_recon_image = 0.85

def SSIM(x, y, scale):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    x = tf.expand_dims(x, axis=0)
    y = tf.expand_dims(y, axis=0)

    mu_x = slim.avg_pool2d(x, scale, 1, 'SAME')
    mu_y = slim.avg_pool2d(y, scale, 1, 'SAME')

    sigma_x = slim.avg_pool2d(x ** 2, scale, 1, 'SAME') - mu_x ** 2
    sigma_y = slim.avg_pool2d(y ** 2, scale, 1, 'SAME') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y, scale, 1, 'SAME') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.squeeze(tf.clip_by_value((1 - SSIM) / 2, 0, 1), axis=0)



def meshgrid(height, width, is_homogeneous=True):
  """Construct a 2D meshgrid.

  Args:
    height: height of the grid
    width: width of the grid
    is_homogeneous: whether to return in homogeneous coordinates
  Returns:
    x,y grid coordinates [height, width, 2 (3 if homogeneous)]
  """
  x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                  tf.transpose(tf.expand_dims(
                      tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
  y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                  tf.ones(shape=tf.stack([1, width])))
  x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
  y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
  if is_homogeneous:
    ones = tf.ones_like(x_t)
    coords = tf.stack([x_t, y_t, ones], axis= -1)
  else:
    coords = tf.stack([x_t, y_t], axis= -1)
  return coords


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
  """Transforms coordinates in the pixel frame to the camera frame.

  Args:
    depth: [height, width]
    pixel_coords: homogeneous pixel coordinates [height, width, 3]
    intrinsics: camera intrinsics [height, width, 3, 3]
    is_homogeneous: return in homogeneous coordinates
  Returns:
    Coords in the camera frame [height, width, 3 (4 if homogeneous)]
  """
  height, width, _ = depth.get_shape().as_list()
  pixel_coords = tf.expand_dims(pixel_coords, axis=-1)
  cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords)[..., 0] * depth
  if is_homogeneous:
    ones = tf.ones([height, width, 1])
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
  cam_coords = tf.expand_dims(cam_coords, axis=-1)
  pixel_coords = tf.matmul(intrinsics, cam_coords)[..., 0]
  # pixel_coords = tf.concat([pixel_coords[..., 0:2] / pixel_coords[..., 2:], pixel_coords[..., 2:]], axis=-1)
  pixel_coords = pixel_coords / (pixel_coords[..., 2:3] + 1e-8)

  return pixel_coords


def shape_deformation(src_coords, R_mat, T_mat):
    src_coords = tf.expand_dims(src_coords, axis=-1)
    dfm_coords = tf.squeeze(tf.matmul(R_mat, src_coords),-1) + T_mat
    return dfm_coords


def compute_nonrigid_flow(depth, R_mat, T_mat, intrinsics):
    height, width, _ = depth.get_shape().as_list()

    # Construct pixel grid coordinates
    pixel_coords = meshgrid(height, width, True)
    # Convert pixel coordinates to the camera frame
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics, False)
    dfm_cam_coords = shape_deformation(cam_coords, R_mat, T_mat)
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
        coords_x, coords_y = tf.split(coords, [1, 1], axis=2)

        img_size_int = imgs.get_shape().as_list()
        coord_size_int = coords.get_shape().as_list()

        coords_y = tf.cast(coords_y, 'float32')
        coords_x = tf.cast(coords_x, 'float32')

        coords_y = coords_y / coord_size_int[0] * img_size_int[0]
        coords_x = coords_x / coord_size_int[1] * img_size_int[1]

        y_max = tf.cast(tf.shape(imgs)[0] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        coords_y = tf.clip_by_value(coords_y, zero, y_max)
        coords_x = tf.clip_by_value(coords_x, zero, x_max)

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        idx00 = tf.concat([y0_safe, x0_safe], axis= -1)
        idx01 = tf.concat([y1_safe, x0_safe], axis= -1)
        idx10 = tf.concat([y0_safe, x1_safe], axis= -1)
        idx11 = tf.concat([y1_safe, x1_safe], axis= -1)

        im00 = tf.gather_nd(imgs, tf.cast(idx00, 'int32'))
        im01 = tf.gather_nd(imgs, tf.cast(idx01, 'int32'))
        im10 = tf.gather_nd(imgs, tf.cast(idx10, 'int32'))
        im11 = tf.gather_nd(imgs, tf.cast(idx11, 'int32'))

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = w00 * im00 + w01 * im01 + w10 * im10 + w11 * im11

    return output

def image_similarity(x, y):
    # return alpha_recon_image * (SSIM(x,y, 3)) + (1-alpha_recon_image) * tf.abs(x-y)
    return tf.abs(x-y)
    # return  tf.clip_by_value(tf.abs(x-y), 0, 0.5)

    # return alpha_recon_image * (SSIM(x,y, 1))


def flow_warp(src_img, tgt_img, src_depth, tgt_depth, R_mat, T_mat, intrinsics):
    height, width, _ = src_depth.get_shape().as_list()
    dfm_pixel_coords = compute_nonrigid_flow(src_depth, R_mat, T_mat, intrinsics)
    warpped_sampled_img = bilinear_sampler(tgt_img, dfm_pixel_coords[..., :2])
    pixel_coords = meshgrid(height, width, True)
    src_sampled_img = bilinear_sampler(src_img, pixel_coords[..., :2])

    return src_sampled_img, warpped_sampled_img, dfm_pixel_coords


def persp_depth_opt(src_img, tgt_img,
                    src_depth, tgt_depth,
                    R_mat, T_mat, input_masks, intrinsics_mat):


    src_sampled_img, warpped_sampled_img, dfm_pixel_coords = flow_warp(src_img, tgt_img, src_depth, tgt_depth, R_mat, T_mat, intrinsics_mat)
    error_map = image_similarity(src_sampled_img, warpped_sampled_img)

    src_sampled_depth, warpped_sampled_depth, dfm_pixel_coords_1 = flow_warp(src_depth, tgt_depth, src_depth, tgt_depth, R_mat, T_mat, intrinsics_mat)
    depth_diff = tf.abs(src_sampled_depth-warpped_sampled_depth)
    depth_consistency_mask = tf.less(depth_diff, 0.50 * tf.ones_like(depth_diff))

    input_masks = tf.expand_dims(input_masks, axis=-1)
    src_sampled_mask, warpped_sampled_mask, dfm_pixel_coords_2 = flow_warp(tf.to_float(input_masks[0]), tf.to_float(input_masks[1]), src_depth, tgt_depth, R_mat, T_mat, intrinsics_mat)
    src_sampled_mask = src_sampled_mask > 0.5
    warpped_sampled_mask = warpped_sampled_mask > 0.5

    final_mask = tf.logical_and(src_sampled_mask, warpped_sampled_mask)
    final_mask = tf.logical_and(final_mask, depth_consistency_mask)
    # masked_error_map = tf.boolean_mask(error_map, input_masks[0])
    # indices = tf.where(input_masks[0])
    # masked_error_map = tf.gather_nd(error_map, indices)
    # photo_loss = tf.reduce_mean(masked_error_map)
    photo_loss = tf.reduce_mean(error_map, axis=-1)


    # variable for debug
    out_debug_dic = {}
    out_debug_dic['src_sampled_img'] = src_sampled_img
    out_debug_dic['warpped_sampled_img'] = warpped_sampled_img
    out_debug_dic['final_mask'] = src_sampled_mask
    out_debug_dic['dfm_pixel_coords'] = dfm_pixel_coords
    # out_debug_dic['masked_error_map'] = masked_error_map
    # out_debug_dic['pixel_coords'] = pixel_coords
    # out_debug_dic['cam_coords'] = cam_coords
    # out_debug_dic['dfm_cam_coords'] = dfm_cam_coords
    # out_debug_dic['dfm_pixel_coords'] = dfm_pixel_coords
    # out_debug_dic['R_mat'] = R_mat
    # out_debug_dic['intrinsics_mat'] = intrinsics_mat
    # out_debug_dic['src_depth'] = src_depth



    return photo_loss, out_debug_dic



def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy


def compute_smooth_loss(disp, img, mask):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    img = tf.image.resize_bilinear(img, (224, 224), name=None)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

    smoothness_x = disp_gradients_x * tf.to_float(mask[:,:,:-1,:])
    smoothness_y = disp_gradients_y * tf.to_float(mask[:,:-1,:,:])

    return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

# def compute_smooth_loss(disp, img, mask):
#     # disp = tf.expand_dims(disp, axis=0)
#
#     mu_disp = slim.avg_pool2d(disp, 3, 1, 'SAME')
#
#
#     return tf.reduce_mean(tf.abs(mu_disp - disp))


