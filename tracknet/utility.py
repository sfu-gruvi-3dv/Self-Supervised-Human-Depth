import numpy as np
import PIL.Image
import cv2
import openmesh
import math

# show image in Jupyter Notebook (work inside loop)
from io import BytesIO 
from IPython.display import display, Image
def show_img_arr(arr, bgr_mode = False):
    if bgr_mode is True:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    im = PIL.Image.fromarray(arr)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))

# show depth array in Jupyter Notebook (work inside loop)
def show_depth_arr(depth_map):
    depth_max = np.max(depth_map)
    depth_min = np.min(depth_map)
    depth_map = (depth_map - depth_min)/(depth_max - depth_min)*255
    show_img_arr(depth_map.astype(np.uint8))

def shift_verts(proc_param, verts, cam):
    img_size = proc_param['img_size']
    cam_s = cam[0]
    cam_pos = cam[1:]
    flength = 500.
    tz = flength / (0.5 * img_size * cam_s)
    trans = np.hstack([cam_pos, tz])
    vert_shifted = verts + trans
    return vert_shifted

def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor

# get silhouette boundingbox
def get_sil_bbox(sil, margin = 0):
    if len(sil.shape)>2:
        sil = sil[:,:,0]
    sil_col = np.sum(sil,1)
    sil_row = np.sum(sil,0)
    y_min = np.argmax(sil_col>0)
    y_max = len(sil_col) - np.argmax(np.flip(sil_col, 0)>0)
    x_min = np.argmax(sil_row>0)
    x_max = len(sil_row) - np.argmax(np.flip(sil_row, 0)>0)
    if margin != 0:
        y_min -= margin
        x_min -= margin
        y_max += margin
        x_max += margin
    return y_min, y_max, x_min, x_max

# come from hmr-src/util/image.py
def scale_and_crop(image, scale, center, img_size):
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_scaled, ((margin, ), (margin, ), (0, )), mode='edge')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param

# compute Rt
def ct2Rt(center, target, up = [0, -1, 0]):
    center = np.array(center)
    target = np.array(target)
    dir_vec = target - center
    dir_vec = dir_vec / np.linalg.norm(dir_vec)
    
    up_vec = np.array(up)
    up_vec = up_vec - np.dot(up_vec, dir_vec)*dir_vec
    up_vec = up_vec / np.linalg.norm(up_vec)
    
    right_vec = np.cross(dir_vec, up_vec)
    right_vec = right_vec / np.linalg.norm(right_vec)
    Rt = np.zeros((3, 4))
    Rt[0, :3] = right_vec
    Rt[1, :3] = up_vec
    Rt[2, :3] = dir_vec
    Rt[:, 3] = -np.dot(Rt[:3, :3], np.transpose(center))
    return Rt


def rotvec2rotmat(rotvec):
    # computes rotation matrix through Rodrigues formula as in cv2.Rodrigues
    rotvec = np.array(rotvec)
    theta = np.linalg.norm(rotvec)
    r = (rotvec / theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2][0], r[1][0]],
                      [r[2][0], 0, -r[0][0]],
                      [-r[1][0], r[0][0], 0]])
    return (cost * np.eye(3) + (1 - cost) * r.dot(r.T) + np.sin(theta) * mat)

# Compose verts and faces to openmesh TriMesh
def make_trimesh(verts, faces, compute_vn = True):
    # if vertex index starts with 1, make it start with 0
    if np.min(faces) == 1:
        faces = np.array(faces)
        faces = faces - 1
    
    # make a mesh
    mesh = openmesh.TriMesh()

    # transfer verts and faces
    for i in range(len(verts)):
        mesh.add_vertex(verts[i])
    for i in range(len(faces)):
        a = mesh.vertex_handle(faces[i][0])
        b = mesh.vertex_handle(faces[i][1])
        c = mesh.vertex_handle(faces[i][2])
        mesh.add_face(a,b,c)

    # compute vert_norms
    if compute_vn is True:
        mesh.request_vertex_normals()
        mesh.update_normals()

    return mesh

# do register using procrustes analysis
from scipy.linalg import orthogonal_procrustes
def register_ps(src_set, tgt_set):    
    # make array
    src_set = np.asarray(src_set)
    tgt_set = np.asarray(tgt_set)
    
    # move to orginal, input set must set the center as the first element
    src_set_t = src_set - src_set[0]
    tgt_set_t = tgt_set - tgt_set[0]
    
    rotmat, _ = orthogonal_procrustes(tgt_set_t, src_set_t)
    tfvec = tgt_set[0] - np.dot(rotmat, src_set[0])
    
    Rt = np.zeros((3, 4))
    Rt[:3, :3] = rotmat
    Rt[:, 3] = tfvec
    
    return Rt

# for resizing Rt map    
def resize_3d_arr(src_arr, tgt_size):
    src_size = src_arr.shape
    if (len(src_size) - len(tgt_size)) == 1:
        tgt_size = tgt_size + (src_size[-1],)
    tgt_arr = np.zeros(tgt_size)
    for tgt_v in range(tgt_size[0]):
        for tgt_u in range(tgt_size[1]):
            src_v = float(tgt_v) / tgt_size[0] * src_size[0]
            src_u = float(tgt_u) / tgt_size[1] * src_size[1]
            src_v = int(np.round(src_v))
            src_u = int(np.round(src_u))
            tgt_arr[tgt_v, tgt_u, :] = src_arr[src_v, src_u, :]
    return tgt_arr

# ncc of patched windowed images
def get_list_ncc(patch_list):
    src_patch = patch_list[0]
    src_patch = src_patch/np.linalg.norm(src_patch)

    ncc_list = []
    for patch in patch_list:
        ncc_list.append(np.dot(src_patch, patch) / np.linalg.norm(patch))
    return np.mean(ncc_list)

# get the position of max/min from a list, if there are max/min elements, --
# -- select the one which are closest to the base number
def get_max_closest_to_base(src_list, base, inv = False):
    if inv is False:
        max_value = max(src_list)
    elif inv is True:
        max_value = min(src_list)
    max_list = [i for i, j in enumerate(src_list) if j == max_value]
    max_dist_list = np.abs(np.asarray(max_list) - base).tolist()
    max_dist_ind = max_dist_list.index(np.min(max_dist_list))
    max_close_ind = max_list[max_dist_ind]
    return max_close_ind

# rotate verts along y axis
def rotate_verts_y(verts, y):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = y*math.pi/180
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])

    for i in range(len(verts)):
        verts[i] = np.dot(R, verts[i])
    verts = verts + verts_mean
    return verts

# rotate verts along x axis
def rotate_verts_x(verts, x):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = x*math.pi/180
    R = np.array([[1, 0, 0],
                  [0, np.cos(angle), -np.sin(angle)],
                  [0, np.sin(angle), np.cos(angle)]])

    for i in range(len(verts)):
        verts[i] = np.dot(R, verts[i])
    verts = verts + verts_mean
    return verts

# rotate verts along z axis
def rotate_verts_z(verts, z):
    verts_mean = np.mean(verts, axis = 0)
    verts = verts - verts_mean

    angle = z*math.pi/180
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])

    for i in range(len(verts)):
        verts[i] = np.dot(R, verts[i])
    verts = verts + verts_mean
    return verts


# smooth triangle mesh in openmesh
def smooth_mesh(src_mesh, iter_num = 3):
    src_verts = src_mesh.points()
    v_v_list = src_mesh.vertex_vertex_indices()
    while(iter_num>0):
        iter_num -= 1
        tgt_verts = []
        for v_list in v_v_list:
            v_list_real = v_list[v_list!=-1]
            v_v_center = np.mean([src_verts[v] for v in v_list_real], 0)
            tgt_verts.append(v_v_center)
        src_verts = tgt_verts
    
    tgt_mesh = make_trimesh(tgt_verts, src_mesh.face_vertex_indices())
    return tgt_mesh

def expand_mesh(src_mesh, expand_len = 0.005):
    src_mesh.request_vertex_normals()
    src_mesh.update_vertex_normals()
    src_verts = src_mesh.points()
    tgt_verts = src_verts + src_mesh.vertex_normals()*expand_len
    tgt_mesh = make_trimesh(tgt_verts, src_mesh.face_vertex_indices())
    return tgt_mesh

def depth2pts(depth, intrinsic):
    h = depth.shape[0]
    w = depth.shape[1]

    xv_np, yv_np = np.meshgrid(np.linspace(0, w, w + 1)[:-1], np.linspace(0, h, h + 1)[:-1], indexing='xy')

    point_x = (xv_np - intrinsic[0, 2]) * depth[:, :] / intrinsic[0, 0]
    point_y = (yv_np - intrinsic[1, 2]) * depth[:, :] / intrinsic[1, 1]

    points = np.stack([point_x, point_y, depth], axis=-1)
    return  points

def rotmatt2rotvect(rotmatt_flat):
    if np.all(rotmatt_flat==0):
        return np.zeros(6, dtype=np.float32)
    rotmatt = np.resize(rotmatt_flat, (3, 4))
    rotvec = cv2.Rodrigues(rotmatt[:3,:3])[0].T[0]
    rotvect = np.concatenate((rotvec, rotmatt[:,3].T))
    return rotvect

def rotvect2rotmatt(rotvectt):
    if np.all(rotvectt==0):
        return np.zeros(12, dtype=np.float32)
    rotmat = cv2.Rodrigues(rotvectt[:3])[0]
    rotmatt = np.concatenate((rotmat, np.expand_dims(rotvectt[3:6], 1)), 1)
    rotmatt_flat = np.ndarray.flatten(rotmatt)
    return rotmatt_flat

def save_Rtmap_png(Rt_map, filename, matrix_flag=True):
    assert Rt_map.shape == (256, 256, 12), "Rt_map shape mismatch"
    if matrix_flag==False:
        rotvect_map = np.apply_along_axis(rotmatt2rotvect, 2, Rt_map)
        rotvect_map = rotvect_map.astype(np.float32)
        rotvect_str = rotvect_map.tostring()
        rotvect_str_uint8 = np.reshape(np.frombuffer(rotvect_str, dtype=np.uint8), (256, 256*6, 4))
    else:
        rotvect_map = Rt_map
        rotvect_map = rotvect_map.astype(np.float32)
        rotvect_str = rotvect_map.tostring()
        rotvect_str_uint8 = np.reshape(np.frombuffer(rotvect_str, dtype=np.uint8), (256, 256*12, 4))

    cv2.imwrite(filename, rotvect_str_uint8)
    return True

def read_Rtmap_png(filename, matrix_flag=True):
    rotvect_str_uint8 = cv2.imread(filename, -1)
    if matrix_flag==False:
        assert rotvect_str_uint8.shape==(256, 256*6, 4), "rotvect_str_uint8 mismatch"    
        rotvect_str2 = rotvect_str_uint8.tostring()
        rotvect_map2 = np.frombuffer(rotvect_str2, dtype=np.float32)
        rotvect_map2 = np.reshape(rotvect_map2, (256, 256, 6))
        rotvect_map2 = np.apply_along_axis(rotvect2rotmatt, 2, rotvect_map2)
    else:
        assert rotvect_str_uint8.shape==(256, 256*12, 4), "rotvect_str_uint8 mismatch"    
        rotvect_str2 = rotvect_str_uint8.tostring()
        rotvect_map2 = np.frombuffer(rotvect_str2, dtype=np.float32)
        rotvect_map2 = np.reshape(rotvect_map2, (256, 256, 12))

    return rotvect_map2


def save_depth_png(depth_map, filename):
    assert depth_map.shape == (256, 256), "depth_map shape mismatch"
    depth_map = depth_map.astype(np.float32)
    depth_str = depth_map.tostring()
    depth_str_uint8 = np.resize(np.frombuffer(depth_str, dtype=np.uint8), (256, 256, 4))
    cv2.imwrite(filename, depth_str_uint8)
    return True

def read_depth_png(filename):
    depth_str_uint8 = cv2.imread(filename, -1)
    assert depth_str_uint8.shape==(256, 256, 4), "rotvect_str_uint8 mismatch"
    depth_str = depth_str_uint8.tostring()
    depth_map = np.frombuffer(depth_str, dtype=np.float32)
    depth_map = np.resize(depth_map, (256, 256, 1))
    return depth_map


def normalize_coeff(all_coeff):
    real_t = all_coeff['smpl_coeff_world'][:3]
    smpl_rot = all_coeff['smpl_coeff_world'][3:6]
    smpl_pose = all_coeff['smpl_coeff_world'][6:75]
    smpl_shape = all_coeff['smpl_coeff_world'][75:85]
    bbx_t = all_coeff['bbx_t']
    global_t = (np.asarray(real_t) + np.asarray(bbx_t)).tolist()
    coeff = np.array(global_t + smpl_rot + smpl_pose + smpl_shape,
                     dtype=np.float32)
    return coeff


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# show image in Jupyter Notebook (work inside loop)
from io import BytesIO 
from IPython.display import display, Image
def show_img_arr(arr, bgr_mode = False):
    if bgr_mode is True:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    im = PIL.Image.fromarray(arr)
    bio = BytesIO()
    im.save(bio, format='png')
    display(Image(bio.getvalue(), format='png'))

# show depth array in Jupyter Notebook (work inside loop)
def show_depth_arr(depth_map):
    depth_max = np.max(depth_map)
    depth_min = np.min(depth_map)
    depth_map = (depth_map - depth_min)/(depth_max - depth_min)*255
    show_img_arr(depth_map.astype(np.uint8))

# Compose verts and faces to openmesh TriMesh
def make_trimesh(verts, faces, compute_vn = True):
    # if vertex index starts with 1, make it start with 0
    if np.min(faces) == 1:
        faces = np.array(faces)
        faces = faces - 1
    
    # make a mesh
    mesh = openmesh.TriMesh()

    # transfer verts and faces
    for i in range(len(verts)):
        mesh.add_vertex(verts[i])
    for i in range(len(faces)):
        a = mesh.vertex_handle(faces[i][0])
        b = mesh.vertex_handle(faces[i][1])
        c = mesh.vertex_handle(faces[i][2])
        mesh.add_face(a,b,c)

    # compute vert_norms
    if compute_vn is True:
        mesh.request_vertex_normals()
        mesh.update_normals()

    return mesh

from scipy.linalg import orthogonal_procrustes
def register_pc_ps(src_set, tgt_set):    
    # make array
    src_set = np.asarray(src_set)
    tgt_set = np.asarray(tgt_set)
    
    src_center = np.mean(src_set, 0)
    tgt_center = np.mean(tgt_set, 0)
    
    # move to orginal, input set must set the center as the first element
    src_set_t = src_set - src_center
    tgt_set_t = tgt_set - tgt_center
    
    rotmat, _ = orthogonal_procrustes(tgt_set_t, src_set_t)
    tfvec = tgt_center - np.dot(rotmat, src_center)
    
    Rt = np.zeros((3, 4))
    Rt[:3, :3] = rotmat
    Rt[:, 3] = tfvec
    
    return Rt


# save using cv2 functions
def save_img_arr(img_arr, filename):
    img_pil = PIL.Image.fromarray(img_arr)
    img_pil.save(filename)
    return True


# render predicted coefficients
import pyrender, trimesh
import sys
sys.path.append("./smpl/smpl_webuser/")
from smpl.smpl_webuser.serialization import load_model
def render_pred_coeff(pred_coeff):

    pred_coeff = np.asarray(pred_coeff).tolist()

    # make scene
    scene = pyrender.Scene()

    # initialize camera
    f_len = 366.474487
    rend_size = 256.
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
    verts_bias = pred_coeff[3:6]
    smpl_op.pose[:] = np.asarray([0]*3 + pred_coeff[6:75])
    smpl_op.betas[:] = np.array(pred_coeff[75:85])
    verts = np.array(smpl_op)
    rot_mat = cv2.Rodrigues(np.asarray(pred_coeff[3:6]))[0]
    verts = np.tensordot(verts, rot_mat, axes=([1],[1]))
    verts = verts + pred_coeff[:3]

    # make trimesh
    faces = np.load("./tf_smpl/smpl_faces.npy").astype(np.int32)
    this_trimesh = trimesh.Trimesh(vertices = verts, faces = faces)
    this_mesh = pyrender.Mesh.from_trimesh(this_trimesh)

    scene.add(this_mesh)
    rend_img, _ = r.render(scene)

    return rend_img



