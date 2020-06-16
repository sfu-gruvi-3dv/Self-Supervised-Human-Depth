import tensorflow as tf
import numpy as np
import cv2



def read_depth_png(filename):
    depth_str_uint8 = cv2.imread(filename, -1)
    depth_str = depth_str_uint8.tostring()
    depth_map = np.frombuffer(depth_str, dtype=np.float32)
    depth_map = np.reshape(depth_map, (256, 256, 1))
    return depth_map


def read_Rtmap_png(filename):
    rotvect_str_uint8 = cv2.imread(filename, -1)
    rotvect_str2 = rotvect_str_uint8.tostring()
    rotvect_map2 = np.frombuffer(rotvect_str2, dtype=np.float32)
    rotvect_map2 = np.reshape(rotvect_map2, (256, 256, 12))
    return rotvect_map2


def read_decode_img(img_dir):
    img = tf.io.read_file(img_dir)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return img

def read_depth_png_tf(depth_dir):
    depth_dir = depth_dir.numpy().decode("utf-8")
    depth = read_depth_png(depth_dir)
    return depth

def read_decode_depth(depth_dir):
    depth = tf.py_function(func=read_depth_png_tf, inp=[depth_dir], Tout=tf.float32)
    return depth

def read_Rtmap_png_tf(RtMap_dir):
    RtMap_dir = RtMap_dir.numpy().decode("utf-8")
    RtMap = read_Rtmap_png(RtMap_dir)
    return RtMap

def read_decode_RtMap(RtMap_dir):
    RtMap = tf.py_function(func=read_Rtmap_png_tf, inp=[RtMap_dir], Tout=tf.float32)
    return RtMap


def read_instrinsic_matrix_tf():
    # intrinsic_mat = np.array((329.7, 0, 96.27,
    #                           0 , 329.7, 63.30,
    #                           0, 0, 1))
    # intrinsic = np.array((442.5352 / 2, 0, 256 / 2,
    #                       0, 442.5352 / 2, 256 / 2,
    #                       0, 0, 1))
    intrinsic = tf.constant([[222.2364, 0, 130.95],
                          [0, 222.2364, 126.53],
                          [0, 0, 1]])
    # intrinsic = tf.constant([[366.4744, 0, 128.00],
    #                       [0, 366.4744, 128.00],
    #                       [0, 0, 1]])

    return intrinsic