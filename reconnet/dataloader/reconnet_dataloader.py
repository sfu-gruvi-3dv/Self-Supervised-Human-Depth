from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import random, glob, json
from PIL import Image
import cv2

# import utils.utils as util
import function.utils as func_utils
from dataloader.utils import read_depth_png, read_Rtmap_png, read_decode_img, read_decode_depth, read_decode_RtMap, read_instrinsic_matrix_tf
import time


class data_loader():

    def color(self, x):
        x = tf.image.random_hue(x, 0.05)
        x = tf.image.random_saturation(x, 0.7, 1.3)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.8, 1.2)
        return x

    def _data_load(self, folder_name):
        folder_name = folder_name.numpy().decode("utf-8")
        tgt_img = np.asarray(Image.open(folder_name + '/0.png').convert('RGB'))

        tgt_depth = np.load(folder_name + '/depth_map_0.npy')[..., np.newaxis]
        smpl_depth = np.load(folder_name + '/depth_map_smpl.npy')[..., np.newaxis]

        if self.eval_flag == True:
            gt_depth = np.load(folder_name + '/depth_map_' + str(0) + '_gt.npy')[..., np.newaxis]


        ref_img_list = []
        RtMap_list = []

        for i in range(self.ref_num):
            ref_img_list.append(np.asarray(Image.open(folder_name + '/' + str(i+1) + '.png').convert('RGB')))
            deformation = np.load(folder_name + '/Rt_map_' + str(i + 1) + '.npy').reshape(
                (self.height // self.divide, self.width // self.divide, 3, 4))

            RtMap_list.append(deformation)

        intrinsics = func_utils.read_instrinsic_matrix()
        ref_imgs = np.stack(ref_img_list)
        RtMaps = np.stack(RtMap_list)


        if self.eval_flag == True:
            return tgt_img / 255.0, ref_imgs / 255.0, tgt_depth, intrinsics, RtMaps, folder_name, smpl_depth, gt_depth
        else:
            return tgt_img / 255.0, ref_imgs / 255.0, tgt_depth, intrinsics, RtMaps, folder_name, smpl_depth


    def _parse_function_np(self, folder_name):
        if self.eval_flag == True:
            data = tf.py_function(func=self._data_load, inp=[folder_name],
                                  Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                        tf.string, tf.float32, tf.float32])

        else:
            data = tf.py_function(func=self._data_load, inp=[folder_name],
                                  Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
                                        tf.string, tf.float32])

        return data

    def _parse_function_tf(self, folder_name):
        tgt_img_dir = tf.string_join([folder_name, tf.constant('/0.png')])
        tgt_img = read_decode_img(tgt_img_dir)

        tgt_depth_dir = tf.string_join([folder_name, tf.constant('/depth_map_0.png')])
        tgt_depth = read_decode_depth(tgt_depth_dir)

        # smpl_depth_dir = tf.string_join([folder_name, tf.constant('/depth_map_smpl.png')])
        # smpl_depth = read_decode_depth(smpl_depth_dir)

        if self.eval_flag == True:
            gt_depth_dir = tf.string_join([folder_name, tf.constant('/depth_map_0_gt.png')])
            gt_depth = read_decode_depth(gt_depth_dir)

        ref_img_list = []
        RtMap_list = []

        for i in range(self.ref_num):
            ref_img_dir = tf.string_join([folder_name, tf.constant('/' + str(i+1) + '.png')])
            ref_img_list.append(read_decode_img(ref_img_dir))

            RtMap_dir = tf.string_join([folder_name, tf.constant('/Rt_map_' + str(i + 1) + '.png')])
            RtMap_list.append(read_decode_RtMap(RtMap_dir))

        intrinsics = read_instrinsic_matrix_tf()

        ref_imgs = tf.stack(ref_img_list, axis=0)
        RtMaps = tf.stack(RtMap_list, axis=0)

        # if self.data_aug==True:
        #     combined_img = tf.concat([tgt_img[tf.newaxis, ...], ref_imgs], axis=0)
        #     aug_data = self.color(combined_img)
        #     aug_data = tf.where(combined_img>0, aug_data, tf.zeros_like(aug_data))
        #     tgt_img = aug_data[0]
        #     ref_imgs = aug_data[1:]


        if self.eval_flag == True:
            return tgt_img, ref_imgs, tgt_depth, intrinsics, RtMaps, folder_name, gt_depth
        else:
            return tgt_img, ref_imgs, tgt_depth, intrinsics, RtMaps, folder_name



    def __init__(self, config = None, eval_flag=False, png_flag = False, data_aug=False):

        self.data_dir = config['data_dir']
        self.global_shuffle_flag = config['global_shuffle_flag']
        self.mode = config['mode']
        self.ref_num = config['ref_num']
        self.height = config['height']
        self.width = config['width']
        self.divide = config['divide']
        self.batch_size = config['batch_size']
        self.eval_flag = eval_flag
        self.png_flag = png_flag
        self.data_aug=data_aug

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
            self.sample_ds = sample_dir_ds.map(self._parse_function_np)
        else:
            self.sample_ds = sample_dir_ds.map(self._parse_function_tf)

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
        next_data[4].set_shape([self.batch_size, self.ref_num, self.height//self.divide, self.width//self.divide, 12])
        data_sample['Rt_Maps'] = next_data[4]
        data_sample['folder_name'] = next_data[5]
        # next_data[6].set_shape([self.batch_size, self.height // self.divide, self.width // self.divide, 1])
        # data_sample['smpl_depth'] = next_data[6]


        if self.eval_flag == True:
            next_data[6].set_shape([self.batch_size, self.height // self.divide, self.width // self.divide, 1])
            data_sample['gt_depth'] = next_data[6]

        return data_sample


if __name__ == "__main__":
    config = {'data_dir': '/media/feitongt/Experiment3/human_depth_cvpr2020/data/test2',
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
        start_time = time.time()
        for i in range(1000):
            value = sess.run(next_data)
            print(value['folder_name'][0])
            c = 0
        print("--- %s seconds ---" % (time.time() - start_time))



