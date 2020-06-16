import numpy as np

def compute_depth_err(pred_depth, gt_depth, mask):
    res = 60
    offset_range = 0.20
    err = np.zeros(res, dtype=np.float32)

    float_mask = mask.astype(np.float32)
    pred_depth = np.multiply(pred_depth, float_mask)
    gt_depth = np.multiply(gt_depth, float_mask)
    median_offset = np.median(pred_depth[mask] - np.median(gt_depth[mask]))
    pnum = np.sum(mask)

    for j in range(res):
        offset = (j - (res / 2)) * (offset_range / res)
        pred_depth_shift = np.multiply((np.copy(pred_depth) + offset + median_offset), float_mask)
        errmap = np.abs(pred_depth_shift - gt_depth)
        err[j] = np.sum(errmap) / pnum

    minloc = np.argmin(err)
    offset = (minloc - (res / 2)) * (offset_range / res)
    depth_shift = np.multiply((np.copy(pred_depth) + offset + median_offset), float_mask)
    errmap = np.absolute(depth_shift - gt_depth)

    return errmap, offset + median_offset