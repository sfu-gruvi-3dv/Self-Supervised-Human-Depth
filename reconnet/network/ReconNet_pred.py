'''
Model for multiview optimization
(c) Feitong, Tan
'''
import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import function.utils as func_utils
import utils.utils as util
from function.dynamic_flow_warp import compute_nonrigid_flow, bilinear_sampler, image_similarity, SSIM, compute_smooth_loss
# from network.hourglass import HourglassModel
from network.hourglass import HourglassModel


class ReconNetTrain(object):
    def __init__(self, config):
        self.config = config
        self.num_feats = config['num_feats']
        self.num_blocks = config['num_blocks']
        self.num_stacks = config['num_stacks']
        self.depth_outDims = config['depth_outDims']
        self.initial_flag = config['initial_flag']

        self.output_dir = config['output_dir']
        self.multiscale_weight = config['multiscale_weight']
        self.img_resolution = config['resolution']
        self.divide = config['divide']
        self.depth_resolution = [config['resolution'][0] // self.divide, config['resolution'][1] // self.divide]
        self.divide = config['divide']
        self.num_scale = 1
        self.batch_size = config['batch_size']
        self.ref_num = config['ref_num']

        self.debug_dic = {}


    def inference(self):

        with tf.variable_scope('placeholders'):
            self.tgt_img_pl = tf.placeholder(tf.float32, [self.batch_size, self.img_resolution[0],
                                                          self.img_resolution[1], 3], name='tgt_img')
            self.tgt_depth_pl = tf.placeholder(tf.float32, [self.batch_size, self.depth_resolution[0],
                                                            self.depth_resolution[1], 1], name='tgt_depth')

        self.tgt_mask_pl = tf.math.greater(self.tgt_depth_pl, 0.1)
        self.tgt_mask_float = tf.to_float(self.tgt_mask_pl)
        self.smpl_mask = tf.math.greater(self.tgt_depth_pl, 0.1)
        self.smpl_mask_float = tf.to_float(self.smpl_mask)

        # self.tgt_img = tf.image.rgb_to_grayscale(self.tgt_img_pl)
        self.tgt_img = self.tgt_img_pl


        self.depth_offset = self.build_depthnet()
        depth = self.tgt_depth_pl + self.depth_offset

        return depth


    def build_depthnet(self):
        with tf.variable_scope('depth_estimation'):
            self.depth_model = HourglassModel(True, self.num_stacks, self.num_feats, self.depth_outDims, self.num_blocks,
                                         name='depth_net')
            self.ds_tgt_img_pl = self.tgt_img

            mean_depth = tf.math.reduce_sum(self.tgt_depth_pl, axis=[1,2,3], keep_dims=True) / tf.math.reduce_sum(self.smpl_mask_float, axis=[1,2,3], keep_dims=True)
            depth_shifted = tf.where(self.smpl_mask, self.tgt_depth_pl - mean_depth, tf.zeros_like(self.tgt_depth_pl))
            depth_shifted = tf.image.resize_nearest_neighbor(depth_shifted,
                                                                  (self.depth_resolution[0]*2, self.depth_resolution[1]*2))
            concat = tf.concat([self.ds_tgt_img_pl, depth_shifted], axis=-1)
            depth_stack = self.depth_model.generate(concat)

            ## sigmoid
            depth_offset_base = (tf.math.sigmoid(depth_stack['baseOut'][0]) - 0.5) * 0.15

            ## base
            # depth_filter = tf.concat((tf.range(-5, 0, 1, dtype=tf.float32), tf.range(1, 6, 1, dtype=tf.float32)), axis=0)
            # depth_filter = tf.reshape(depth_filter, [1, 1, 10, 1])
            #
            # depth_logits_base = tf.nn.softmax(depth_stack['baseOut'][0])
            # depth_offset_base = tf.nn.conv2d(depth_logits_base, depth_filter, strides=[1, 1, 1, 1], padding='SAME', name='depth') * 0.015

            # ## detail
            # depth_logits_detail = tf.nn.softmax(depth_stack['detailOut'][0])
            # depth_offset_detail = tf.nn.conv2d(depth_logits_detail, depth_filter, strides=[1, 1, 1, 1], padding='SAME', name='depth') * 0.015

            return depth_offset_base


    def save_model(self, global_variables):
        saver = tf.train.Saver(global_variables, max_to_keep=30)
        self.saver = saver
        return saver

    def load_model(self, model_dir = '/', sess=tf.Session()):
        init = tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        restore_dir = model_dir
        if restore_dir:
            global_vars = tf.global_variables()
            vars_in_checkpoint, _ = func_utils.scan_checkpoint_for_vars(restore_dir, global_vars)
            saver_restore_ckpt = tf.train.Saver(vars_in_checkpoint)
            saver_restore_ckpt.restore(sess, restore_dir)

            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            print('restore succeed')
        else:
            print('Initializing')
            sess.run(init)
            print('Initialization succeed')



