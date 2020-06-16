from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import random, glob, json
from PIL import Image
import cv2

import utils.utils as util
import function.utils as func_utils
from dataloader.utils import read_depth_png, read_Rtmap_png


class data_loader():

    def _data_load(self, folder_name):
        folder_name = folder_name.numpy().decode("utf-8")
        tgt_img = np.asarray(Image.open(folder_name + '/' + str(0) + '.png').convert('RGB'))

        if self.png_flag == False:
            tgt_depth = np.load(folder_name + '/depth_map_' + str(0) + '.npy')[..., np.newaxis]
            smpl_depth = np.load(folder_name + '/depth_map_smpl.npy')[..., np.newaxis]
        else:
            tgt_depth = read_depth_png(folder_name + '/depth_map_' + str(0) + '.png')
            smpl_depth = read_depth_png(folder_name + '/depth_map_smpl.png')

        if self.eval_flag == True:
            if self.png_flag == False:
                gt_depth = np.load(folder_name + '/depth_map_' + str(0) + '_gt.npy')[..., np.newaxis]
            else:
                gt_depth = read_depth_png(folder_name + '/depth_map_' + str(0) + '_gt.png')

        ref_img_list = []
        R_list = []
        t_list = []

        for i in range(self.ref_num):
            ref_img_list.append(np.asarray(Image.open(folder_name + '/' + str(i+1) + '.png').convert('RGB')))
            if self.png_flag == False:
                deformation = np.load(folder_name + '/Rt_map_' + str(i + 1) + '.npy').reshape(
                    (self.height // self.divide, self.width // self.divide, 3, 4))
            else:
                deformation = read_Rtmap_png(folder_name + '/Rt_map_' + str(i + 1) + '.png').reshape(
                    (self.height // self.divide, self.width // self.divide, 3, 4))

            R, t = func_utils.split_Rt(deformation)
            R_list.append(R)
            t_list.append(t)

        intrinsics = func_utils.read_instrinsic_matrix()
        ref_imgs = np.stack(ref_img_list)
        R = np.stack(R_list)
        t = np.stack(t_list)

        # kernel = np.ones((3, 3), np.uint8)
        # smpl_depth_erosion = cv2.erode(smpl_depth, kernel, iterations=2)
        mask = tgt_depth > 0

        if self.eval_flag == True:
            return tgt_img / 255.0, ref_imgs / 255.0, tgt_depth, intrinsics, R, t, mask, folder_name, smpl_depth, gt_depth
        else:
            return tgt_img / 255.0, ref_imgs / 255.0, tgt_depth, intrinsics, R, t, mask, folder_name, smpl_depth


    def _parse_function_np(self, folder_name):
        if self.eval_flag == True:
            data = tf.py_function(func=self._data_load, inp=[folder_name],
                                  Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.bool,
                                        tf.string, tf.float32, tf.float32])

        else:
            data = tf.py_function(func=self._data_load, inp=[folder_name],
                                  Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.bool,
                                        tf.string, tf.float32])

        return data


    def __init__(self, config = None, eval_flag=False, smpl_depth_flag = False):

        self.data_dir = config['data_dir']
        self.global_shuffle_flag = config['global_shuffle_flag']
        self.mode = config['mode']
        self.ref_num = config['ref_num']
        self.height = config['height']
        self.width = config['width']
        self.divide = config['divide']
        self.batch_size = config['batch_size']
        self.smpl_depth_flag = smpl_depth_flag
        self.eval_flag = eval_flag
        self.png_flag = True

        if self.mode == 'train':
            sample_dir_list = glob.glob(self.data_dir + '/*/0*')
            self.sample_dir_list = np.sort(sample_dir_list)
            sample_num = len(self.sample_dir_list)
            print("dataloader: %d sequences detected." % sample_num)

        if self.global_shuffle_flag == True and eval_flag == False:
            random.shuffle(self.sample_dir_list)

        # make datasets
        if eval_flag == True:
            sample_dir_ds = tf.data.Dataset.from_tensor_slices(self.sample_dir_list)
        else:
            sample_dir_ds = tf.data.Dataset.from_tensor_slices(self.sample_dir_list).shuffle(buffer_size=sample_num).repeat()

        if self.png_flag == False:
            self.sample_ds = sample_dir_ds.map(self._parse_function_np, num_parallel_calls=-1)
        else:
            self.sample_ds = sample_dir_ds.map(self._parse_function, num_parallel_calls=-1)

    def get_iterator(self):
        ds = self.sample_ds.batch(self.batch_size).prefetch(buffer_size=30)
        iterator = ds.make_one_shot_iterator()
        next_data = iterator.get_next()
        data_sample = {}
        next_data[0].set_shape([self.batch_size, self.height, self.width, 3])
        data_sample['tgt_img'] = next_data[0]
        next_data[1].set_shape([self.batch_size, self.ref_num, self.height, self.width, 3])
        data_sample['ref_imgs'] = next_data[1]
        next_data[2].set_shape([self.batch_size, self.height//self.divide, self.width//self.divide, 1])
        data_sample['tgt_depth'] = next_data[2]
        next_data[3].set_shape([self.batch_size, 3, 3])
        data_sample['intrinsics'] = next_data[3]
        next_data[4].set_shape([self.batch_size, self.ref_num, self.height//self.divide, self.width//self.divide, 3, 3])
        data_sample['R'] = next_data[4]
        next_data[5].set_shape([self.batch_size, self.ref_num, self.height//self.divide, self.width//self.divide, 3])
        data_sample['t'] = next_data[5]
        next_data[6].set_shape([self.batch_size, self.height//self.divide, self.width//self.divide, 1])
        data_sample['tgt_mask'] = next_data[6]
        data_sample['folder_name'] = next_data[7]
        next_data[8].set_shape([self.batch_size, self.height // self.divide, self.width // self.divide, 1])
        data_sample['smpl_depth'] = next_data[8]

        if self.eval_flag == True:
            next_data[9].set_shape([self.batch_size, self.height // self.divide, self.width // self.divide, 1])
            data_sample['gt_depth'] = next_data[9]

        return data_sample


if __name__ == "__main__":
    config = {'data_dir': '/media/feitongt/Experiment3/human_depth_cvpr2020/data/test',
              'ref_num':6,
              'height': 512,
              'width': 512,
              'divide': 2,
              'global_shuffle_flag': True,
              'mode': 'train',
              'batch_size': 4}
    reconnet_dataloader = data_loader(config = config)
    next_data = reconnet_dataloader.get_iterator()
    with tf.Session() as sess:
        for i in range(100):
            value = sess.run(next_data)
            c = 0



