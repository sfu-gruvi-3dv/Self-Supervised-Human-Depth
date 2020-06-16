# -*- coding: utf-8 -*-
"""
author: Kel

original author: Walid Benbihi1
"""

import tensorflow as tf


class HourglassModel():

    def __init__(self, training, nStack=2, nFeat=256, outDim=16, nLow=4, name='stacked_hourglass', reuse=False):
        """
            args:
                nStack     : number of stacks of (hourglass block)
                nFeat      : number of features in each block
                outDim     : number of output dimension (16 for body joints, 15 for 2D segmentation)
                nLow       : how many times of downsampling in hourglass block
                train      : for batch normalization
        """
        self.nStack = nStack
        self.name = name
        self.nFeat = nFeat
        self.outDim = outDim
        self.train = training
        self.nLow = nLow

    def generate(self, inputs):
        with tf.variable_scope(self.name):
            with tf.variable_scope('preprocessing'):
                cnv1 = self.conv_bn_relu(inputs, 64, 5, 2, 'SAME', name='256to128')
                r1 = self.residual(cnv1, self.nFeat, name='r1')
                # pool = tf.layers.average_pooling2d(r1, 2, 2, 'same')
                # pool = tf.contrib.layers.max_pool2d(r1, [2, 2], [1, 1], padding='SAME')
                r4 = self.residual(r1, self.nFeat, name='r4')
                r5 = self.residual(r4, self.nFeat, name='r5')

            output = {}
            base_out = [None] * self.nStack
            detail_out = [None] * self.nStack

            with tf.variable_scope('stacks'):
                inter = r5
                for i in range(self.nStack):
                    with tf.variable_scope('hourglass_' + str(i)):
                        hg = self.hourglass(inter, self.nLow, self.nFeat)
                        ll = self.residual(hg, self.nFeat, name='ll_res')
                        ll = self.residual(ll, self.nFeat, 'base_res')
                        ll = self.residual(ll, self.nFeat, 'base_res1')
                        ll = self.residual(ll, self.nFeat, 'base_res2')

                        base = self.conv_bn_relu(ll, 64, 1, 1,name='base_ll')
                        base = self.conv2d(base, self.outDim, 1, 1, 'SAME', 'baseOut')

                        # # pool = tf.contrib.layers.max_pool2d(ll, [3, 3], [1, 1], padding='SAME')
                        # ll = self.residual(ll, self.nFeat, name='detail_res')
                        # # ll = self.residual(ll, self.nFeat, name='detail_res2')
                        # ll = self.conv_bn_relu(ll, 64, 1, 1,name='detail_ll')
                        # detail = self.conv2d(ll, self.outDim, 1, 1, 'SAME', 'detailOut')
                        base_out[i] = base
                        # detail_out[i] = detail

                        # if i < self.nStack - 1:
                        #     ll_ = self.conv2d(ll, self.nFeat, name='ll_')
                        #     tmpOut_ = self.conv2d(tmpOut, self.nFeat, name='tmpOut_')
                        #     inter = inter + ll_ + tmpOut_

                output['baseOut'] = tf.stack(base_out, name='base_out')
                # output['detailOut'] = tf.stack(detail_out, name='detail_out')
                return output

    def conv2d(self, inputs, filters, kernel_size=1, strides=1, pad='SAME', name='conv2d'):
        """
        Typical conv2d layer
        Notice that BN has its own bias term and conv layer before bn does not need bias term.
        However, the bias here will not matter in that case
        """
        with tf.variable_scope(name):
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            W = tf.get_variable("W", shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                regularizer=regularizer,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable("b", shape=filters, initializer=tf.constant_initializer(0.1), regularizer=regularizer)
            conv = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return tf.add(conv, b, 'conv2d_out')

    def bn_relu(self, inputs, scope='bn_relu'):
        """
        bn -> relu
        """
        norm = tf.contrib.layers.group_norm(inputs, groups=64, epsilon=1e-6, activation_fn=tf.nn.relu, scope=scope)
        return norm

    def conv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='SAME', name='conv_bn_relu'):
        """
           conv -> bn -> relu
        """
        with tf.variable_scope(name):
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            W = tf.get_variable("W", shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                regularizer=regularizer,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return self.bn_relu(conv, scope='bn_relu')

    def deconv_bn_relu(self, inputs, filters, kernel_size=1, strides=1, pad='SAME', name='deconv_bn_relu'):
        with tf.variable_scope(name):
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
            W = tf.get_variable("W", shape=[kernel_size, kernel_size, filters, inputs.get_shape().as_list()[3]],
                                regularizer=regularizer,
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            deconv = tf.nn.conv2d_transpose(inputs, W, [inputs.get_shape().as_list()[0], inputs.get_shape().as_list()[1]*2, inputs.get_shape().as_list()[2]*2, filters], strides=strides, padding='SAME')
            return self.bn_relu(deconv, scope='bn_relu')


    # def conv_bn_relu6(self, inputs, filters, kernel_size=1, strides=1, pad='SAME', name='conv_bn_relu6'):
    #     """
    #        conv -> bn -> relu
    #     """
    #     with tf.variable_scope(name):
    #         regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    #         W = tf.get_variable("W", shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
    #                             regularizer=regularizer,
    #                             initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    #         conv = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding=pad, data_format='NHWC')
    #         return tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, is_training=self.train,
    #                                             activation_fn=tf.nn.relu6, scale=True)

    def convBlock(self, inputs, numOut, name='convBlock'):
        """
        Convolutional Block
        bn -> relu -> conv(1, 1, numIn, numOut/2)->
        bn -> relu -> conv(3, 3, numOut/2, numOut/2)->
        bn -> relu -> conv(1, 1, numOut/2, numOut)
        """
        with tf.variable_scope(name):
            norm_1 = self.bn_relu(inputs, 'bn_relu_1')
            conv_1 = self.conv2d(norm_1, int(numOut / 2), 1, 1, name='conv_1')

            norm_2 = self.bn_relu(conv_1, 'bn_relu_2')
            conv_2 = self.conv2d(norm_2, int(numOut / 2), 3, 1, 'SAME', name='conv_2')

            norm_3 = self.bn_relu(conv_2, 'bn_relu_3')
            conv_3 = self.conv2d(norm_3, int(numOut), 1, 1, name='conv_3')
            return conv_3

    def skipLayer(self, inputs, numOut, name='skipLayer'):
        """
        Skip if number of input channel == numOut,
        otherwise use 1x1 conv to remap the channels to a desired number
        """
        with tf.variable_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self.conv2d(inputs, numOut, 1, 1, 'SAME', name='skipLayer_conv')
                return conv

    def residual(self, inputs, numOut, name='residual'):
        """
        Residual Block
        One path to convBlock, the other to skip layer, then sum
        """
        with tf.variable_scope(name):
            convb = self.convBlock(inputs, numOut)
            skip = self.skipLayer(inputs, numOut)
            return tf.add(convb, skip, 'residual_out')

    def hourglass(self, inputs, n, numOut, name='hourglass'):
        """
        Hourglass Block
        """
        with tf.variable_scope(name):
            inputs = self.residual(inputs, numOut, name='input')
            up_1 = self.residual(inputs, numOut, name='up1')
            # up_1 = self.residual(up_1, numOut, name='up1_1')
            # up_1 = self.residual(up_1, numOut, name='up1_2')
            # up_1 = self.residual(up_1, numOut, name='up1_3')
            # low_ = tf.contrib.layers.max_pool2d(inputs, [2, 2], [2, 2], 'same')
            # low_ = tf.layers.average_pooling2d(inputs, 3, 2, 'same')
            low_ = self.conv_bn_relu(up_1, numOut, 5, 2, 'SAME', name='256to128')
            low_1 = self.residual(low_, numOut, name='low1')
            low_1 = self.residual(low_1, numOut, name='low1_1')
            low_1 = self.residual(low_1, numOut, name='low1_2')
            low_1 = self.residual(low_1, numOut, name='low1_3')
            low_1 = self.residual(low_1, numOut, name='low1_4')
            low_1 = self.residual(low_1, numOut, name='low1_5')
            low_1 = self.residual(low_1, numOut, name='low1_6')
            low_1 = self.residual(low_1, numOut, name='low1_7')
            low_1 = self.residual(low_1, numOut, name='low1_8')
            # low_1 = self.residual(low_1, numOut, name='low1_9')
            # low_1 = self.residual(low_1, numOut, name='low1_10')
            # low_1 = self.residual(low_1, numOut, name='low1_11')



            if n > 1:
                low_2 = self.hourglass(low_1, n - 1, numOut, name='low2')
            else:
                low_2 = self.residual(low_1, numOut, name='low2')
            low_3 = self.residual(low_2, numOut, name='low3')
            up_2 = self.deconv_bn_relu(low_3, numOut, 5, strides=2, pad='SAME', name='deconv')
            # up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(up_1)[1:3], name='upsampling')
            # if n > 4:
            #     return up_2
            return tf.add(up_1, up_2, 'hourglass_out')
            # concat = tf.concat([up_1, up_2], axis=-1, name='concat')
            # concat = self.residual(concat, numOut, name='concat_out')
            # return concat

