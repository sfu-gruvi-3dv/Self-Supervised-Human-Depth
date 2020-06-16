'''
Model for multiview optimization
(c) Feitong, Tan
'''
import argparse
import glob
import os, sys
import random


import cv2
import numpy as np
import tensorflow as tf
import copy
from PIL import Image


sys.path.append("./../.")
import function.utils as func_utils
import utils.utils as util
from network.ReconNet_pred import ReconNetTrain



# random.seed(4000)
np.random.seed(6666)
tf.set_random_seed(6666)
random.seed(6666)

map_type = cv2.COLORMAP_JET
trunc_val = 0.1

height = 512
width = 512
divide = 2
ref_num = 10

learning_rate = 0.00015
nEpochs = 1
iter_by_epoch = 10000
color_norm_factor = 255
batch_size = 1

error_map_flag = True
generate_depth_flag = True

intrinsic = np.array((500, 0, 128.00,
                      0, 500, 128.00,
                      0, 0, 1)).reshape((3,3))


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='./../self_human_depth_model/finetuned', help='model folder (str, default: "/results/")')
    parser.add_argument('--input_dir', default='./../../pred_base_depth', help='input data folder')
    parser.add_argument('--output_dir', default='./../../pred_final_depth')
    args = parser.parse_args()
    print(args)

    output_dir = args.output_dir + '/'
    input_dir = args.input_dir + '/'
    model_dir = args.model_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reconnet_config = {'num_feats': 256, 'num_blocks': 4, 'num_stacks': 1, 'depth_outDims': 1,
              'initial_flag': False, 'output_dir': output_dir, 'multiscale_weight': [1, 1, 1, 1, 1], 'divide': divide,
              'resolution': (height, width), 'debug_flag': False, 'batch_size': batch_size, 'ref_num': ref_num}


    ReconNet = ReconNetTrain(reconnet_config)

    depth = ReconNet.inference()

    ReconNet.save_model(tf.global_variables())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ReconNet.load_model(model_dir=model_dir, sess=sess)

    image_list = np.sort(glob.glob(input_dir + '/*.jpg'))
    depth_list = np.sort(glob.glob(input_dir + '/*.npy'))

    with sess.as_default():
        for idx, image_dir in enumerate(image_list):

            image_dir = image_list[idx]
            depth_dir = depth_list[idx]

            tgt_img = np.asarray(Image.open(image_dir).convert('RGB')) / 255.0
            tgt_depth = np.load(depth_dir)[np.newaxis, ..., np.newaxis]
            tgt_mask = cv2.resize(tgt_depth[0], (height, width), interpolation=cv2.INTER_NEAREST) > 0
            tgt_img[np.logical_not(tgt_mask)] = 0
            tgt_mask_256 = tgt_depth > 0

            tgt_img = tgt_img[np.newaxis, ...]

            pred_depth = sess.run(
                depth, feed_dict={ReconNet.tgt_img_pl: tgt_img,
                           ReconNet.tgt_depth_pl: tgt_depth,
                           ReconNet.tgt_mask_pl: tgt_mask_256
                           })

            pred_depth = pred_depth[0]

            # tgt_img = Image.fromarray((tgt_img[0] * 255).astype(np.uint8))
            # tgt_img.save(output_dir + '/' + os.path.split(image_dir)[1][:-4] + "_final_depth.jpg")
            np.save(output_dir + '/' + os.path.split(image_dir)[1][:-4] + "_final_depth.npy", pred_depth)

            pred_points = util.depth2meshPersp(pred_depth[:, :, 0], tgt_mask_256[0, :, :, 0], intrinsic, output_dir + '/' + os.path.split(image_dir)[1][:-4] + "_final_depth")
            pred_points = util.depth2meshPersp(tgt_depth[0, :, :, 0], tgt_mask_256[0, :, :, 0], intrinsic, output_dir + '/' + os.path.split(image_dir)[1][:-4] + "_base_depth")
            # func_utils.depth2mesh(pred_depth, tgt_mask[0,:,:,0],  eval_dir + image_dir[-7:-4])








