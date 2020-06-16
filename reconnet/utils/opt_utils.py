#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  23/10/18 7:23 PM
#  feitongt
#  opt_utils.py

import tensorflow as tf


def scale_pyramid(img, num_scales):
  if img == None:
    return None
  else:
    scaled_imgs = [img]
    _, h, w, _ = img.get_shape().as_list()
    for i in range(num_scales - 1):
      ratio = 2 ** (i + 1)
      nh = int(h / ratio)
      nw = int(w / ratio)
      scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
    return scaled_imgs

def seperate_RT(deform_matrix):
  R = deform_matrix[:, :, :3, :3]
  T = deform_matrix[:, :, :3, 3]
  return R, T
