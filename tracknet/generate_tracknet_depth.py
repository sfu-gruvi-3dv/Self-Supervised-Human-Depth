# from test_finetuned_track_net.ipynb
import sys, os, cv2, trimesh, argparse, glob
from time import time, strftime, sleep
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from tqdm import trange
from tf_smpl.batch_smpl import SMPL
from tf_smpl import projection as proj_util
from model import Encoder_resnet
from model import Encoder_fc3_dropout
from utility import bcolors
from utility import save_img_arr
from utility import show_img_arr
from utility import show_depth_arr
from utility import make_trimesh
from utility import smooth_mesh
from utility import expand_mesh
sys.path.append("./smpl/smpl_webuser/")
from smpl.smpl_webuser.serialization import load_model

# configure parameters, will be sent to config.ini or argparse
# in the future
# ===========================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='./finetuned_hmr_model/finetuned_hmr_model')
parser.add_argument('--input_dir', default='./../test_img')
parser.add_argument('--output_dir', default='./../pred_base_depth')
args = parser.parse_args()

restore_dir = args.model_dir
tgt_dir = args.input_dir
output_dir = args.output_dir

if os.path.isdir(output_dir) is False:
    os.mkdir(output_dir)

is_training = False
enable_gpu = False
vis_smpl = True
save_img = True
server_mode = False
disable_tf_warn = True

import pyrender
if server_mode is True:
    # enable this to work on a multi-gpu server
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"
    
    # enable this to work on a headless server, otherwise use glet by default
    os.environ["PYOPENGL_PLATFORM"] = "egl" 

if is_training is True:
    batch_size = 20
else:
    batch_size = 1

# disable warnings
if disable_tf_warn is True:
    import logging
    logging.getLogger('tensorflow').disabled = True

num_stage = 3
num_theta = 72
coeff_num = 85

smpl_neutral_path = "./tf_smpl/neutral_smpl_with_cocoplus_reg.pkl"
joint_type = "cocoplus"

f_len = 366.474487
rend_size = 256.
intrinsic = np.array([[f_len/480*256, 0, rend_size/2],
                      [0, f_len/480*256, rend_size/2],
                      [0, 0, 1]])
        
# some predefined functions
# ===========================================================

smpl = SMPL(smpl_neutral_path, joint_type=joint_type)

# render predicted coefficients - hmr
def render_hmr_smpl(hmr_coeff, f_len = 500., rend_size = 224., 
                    req_model = False):
    # hmr_coeff is a 85-vector, named as theta in hmr
    hmr_coeff = np.asarray(hmr_coeff).tolist()

    # make scene
    scene = pyrender.Scene()
    
    # initialize camera    
    camera = pyrender.PerspectiveCamera(yfov=np.arctan(rend_size*0.5/f_len)*2, aspectRatio=1)
    camera_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    scene.add(camera, pose=camera_pose)
    
    # initialize light
    light_posi1 = np.array([[1.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, -1.0],
                            [0.0, 1.0, 0.0, -2.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light_posi2 = np.array([[1.0, 0.0, 0.0, -1.0],
                            [0.0, 0.0, 1.0, -1.0],
                            [0.0, 1.0, 0.0, -2.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light_posi3 = np.array([[1.0, 0.0, 0.0, 1.0],
                            [0.0, 0.0, 1.0, 1.0],
                            [0.0, 1.0, 0.0, -2.0],
                            [0.0, 0.0, 0.0, 1.0]])
    light = pyrender.SpotLight(color=np.array([0.65098039, 0.74117647, 0.85882353]), 
                               intensity=100,
                               innerConeAngle=np.pi/16.0,
                               outerConeAngle=np.pi/6.0)
    scene.add(light, pose=light_posi1)
    scene.add(light, pose=light_posi2)
    scene.add(light, pose=light_posi3)
    
    # get renderer
    r = pyrender.OffscreenRenderer(viewport_width=rend_size, viewport_height=rend_size)
    
    # get verts from smpl coefficients
    smpl_op = load_model("./tf_smpl/neutral_smpl_with_cocoplus_reg.pkl")
    smpl_op.pose[:] = np.asarray(hmr_coeff[3:75])
    smpl_op.betas[:] = np.array(hmr_coeff[75:85])
    verts = np.array(smpl_op)
    global_t = np.array([hmr_coeff[1], hmr_coeff[2], 
                         f_len/(0.5 * rend_size * hmr_coeff[0])])
    verts = verts + global_t    
    faces = np.load("./tf_smpl/smpl_faces.npy").astype(np.int32)
    
    # smooth and expand
    om_mesh = make_trimesh(verts, faces)
    om_mesh = smooth_mesh(om_mesh, 4)
    om_mesh = expand_mesh(om_mesh, 0.026)

    this_trimesh = trimesh.Trimesh(vertices = om_mesh.points(), faces = om_mesh.face_vertex_indices())
    this_mesh = pyrender.Mesh.from_trimesh(this_trimesh)
    
    scene.add(this_mesh)
    rend_img, depth = r.render(scene)
    
    if req_model is True:
        return rend_img, verts, faces, depth
    else:
        return rend_img

# ===========================================================

mean_var = tf.Variable(tf.zeros((1, coeff_num)), name="mean_param", dtype=tf.float32)
theta_prev = tf.tile(mean_var, [batch_size, 1])

print(bcolors.HEADER + "[%s]" % datetime.now() + bcolors.ENDC, end='')
print('Creating Model ... ', end='')
t_start = time()

# make model
src_img_pl = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
gt_coeff_pl = tf.placeholder(tf.float32, shape=(None, coeff_num))

# Extract image features.
img_enc_fn = Encoder_resnet
threed_enc_fn = Encoder_fc3_dropout
loss_coeff, loss_3d = [], []
img_feat, _ = img_enc_fn(src_img_pl,
                         is_training=is_training,
                         reuse=False)

# Main IEF loop
all_verts = []
all_kps = []
all_cams = []
all_Js = []
final_thetas = []
for i in np.arange(num_stage):
    # print('Iteration %d' % i)
    
    state = tf.concat([img_feat, theta_prev], 1)
    if i == 0:
        delta_theta, threeD_var = threed_enc_fn(
            state,
            num_output=coeff_num,
            is_training=is_training,
            reuse=False)
    else:
        delta_theta, _ = threed_enc_fn(
            state,
            num_output=coeff_num,
            is_training=is_training,
            reuse=True)

    # Compute new theta
    theta_here = theta_prev + delta_theta
    
    # Finally update to end iteration.
    theta_prev = theta_here

# make saver
saver = tf.train.Saver()

if enable_gpu is True:
    config = tf.ConfigProto()
else:
    config = tf.ConfigProto(device_count = {'GPU': 0})        
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:  
    
    saver.restore(sess, restore_dir)
    print("Restoring Done (%.03fs)" % (time() - t_start))  # restoring model
    sleep(1)

    imgs_dir = glob.glob(tgt_dir + '/*.jpg')
    imgs_dir.sort()
    for img_dir in imgs_dir:
        src_img_ori = cv2.imread(img_dir)
        src_img_ori = src_img_ori[:,:,:3]
        src_img = cv2.cvtColor(src_img_ori, cv2.COLOR_BGRA2RGB)

        src_img = cv2.resize(src_img, (224, 224), interpolation=cv2.INTER_LINEAR)

        src_img_norm = src_img.astype(np.float)/127.5 - 1.0
        src_img_batch = np.expand_dims(src_img_norm, 0)

        #src_img_batch = np.zeros((1, 256, 256, 3)) # 1x256x256x3 array
        gt_coeff_batch = np.zeros((1, 85)) # 1x85 array
        pred_coeff = \
            sess.run(theta_here,
                     feed_dict={src_img_pl: src_img_batch,
                                gt_coeff_pl: gt_coeff_batch})
        
        pred_img, _, _, pred_depth = render_hmr_smpl(pred_coeff[0], 
                                                     req_model=True)

        src_img_s = cv2.resize(src_img_ori, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_dir + '/' + os.path.split(img_dir)[1][:-4] + '_resize.jpg', src_img_s)


        # pred_depth = cv2.resize(pred_depth, (256, 256), interpolation=cv2.INTER_NEAREST)
        pred_depth_expand = pred_depth.copy()
        for x in range(1, 223):
            for y in range(1, 223):
                if pred_depth[x, y] != 0:
                    continue
                list_win = np.ndarray.flatten(pred_depth[x-1:x+2, y-1:y+2]).tolist()
                list_win = list(filter((0.0).__ne__, list_win))
                if len(list_win) == 0:
                    continue
                pred_depth_expand[x,y] = np.mean(list_win)

        pred_depth_scnear = cv2.resize(pred_depth, (256, 256), interpolation=cv2.INTER_NEAREST)
        pred_depth_sclin = cv2.resize(pred_depth_expand, (256, 256), interpolation=cv2.INTER_LINEAR)
        pred_depth_sclin[pred_depth_scnear==0] = 0
        
        np.save(output_dir + '/' + os.path.split(img_dir)[1][:-4] + "_depth.npy", pred_depth_sclin)

        # exit()


#     for test_num in trange(64):
#         src_img = cv2.imread("./final_test_iccv2019/norm_gt/%04d_rgb.png" % test_num)
#         src_img = src_img[:,:,:3]
#         src_img = cv2.cvtColor(src_img, cv2.COLOR_BGRA2RGB)
        
#         src_img_norm = src_img.astype(np.float)/127.5 - 1.0
#         src_img_batch = np.expand_dims(src_img_norm, 0)
        
#         #src_img_batch = np.zeros((1, 256, 256, 3)) # 1x256x256x3 array
#         gt_coeff_batch = np.zeros((1, 85)) # 1x85 array
#         pred_coeff = \
#             sess.run(theta_here,
#                      feed_dict={src_img_pl: src_img_batch,
#                                 gt_coeff_pl: gt_coeff_batch})
        
#         pred_img, _, _, pred_depth = render_hmr_smpl(pred_coeff[0], 
#                                                      req_model=True)

#         src_img_s = ((src_img_batch[0] + 1) * 127.5).astype(np.uint8)
#         #show_img_arr(src_img_s)
#         #show_depth_arr(pred_depth)
#         src_img_s = cv2.cvtColor(src_img_s, cv2.COLOR_RGB2BGR)
#         src_img_s = cv2.resize(src_img_s, (256, 256), interpolation=cv2.INTER_LINEAR)
#         cv2.imwrite("./final_test_iccv2019/my_pred_ft4000_hmr/%04d_img.png" % test_num, src_img_s)
        
#         #pred_depth = cv2.resize(pred_depth, (256, 256), interpolation=cv2.INTER_NEAREST)
#         pred_depth_expand = pred_depth.copy()
#         for x in range(1, 223):
#             for y in range(1, 223):
#                 if pred_depth[x, y] != 0:
#                     continue
#                 list_win = np.ndarray.flatten(pred_depth[x-1:x+2, y-1:y+2]).tolist()
#                 list_win = list(filter((0.0).__ne__, list_win))
#                 if len(list_win) == 0:
#                     continue
#                 pred_depth_expand[x,y] = np.mean(list_win)

#         pred_depth_scnear = cv2.resize(pred_depth, (256, 256), interpolation=cv2.INTER_NEAREST)
#         pred_depth_sclin = cv2.resize(pred_depth_expand, (256, 256), interpolation=cv2.INTER_LINEAR)
#         pred_depth_sclin[pred_depth_scnear==0] = 0
        
#         np.save("./final_test_iccv2019/my_pred_ft_hmr/%04d_depth.npy" % test_num, pred_depth_sclin)
        
# print("All done")
