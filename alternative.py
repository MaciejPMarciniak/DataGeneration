import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_dir = '/home/wj17/Downloads/echonetexample.xlsx'
pd = pd.read_excel(data_dir, sheet_name = 'Sheet1')
framenum = 61
frame41 = pd[pd.Frame==framenum] # 61


x1 = frame41.X1.values
y1 = frame41.Y1.values
x2 = frame41.X2.values
y2 = frame41.Y2.values


all_pts = np.concatenate((np.array(list(zip(x1,y1))), (np.array(list(zip(x2,y2))))))


def get_special_pts(pts, show):
    """
    From Maciej's contours, first point is apex.
    Mid point is basal center.
    Find top points as ones that maximize distance from apex (vd)
    and maximize distance from basal center (hd)
    """
    num_pts = pts.shape[0]
    basal_center_idx = int(num_pts/2)
    basal_center = all_pts[basal_center_idx]
    apex = all_pts[0]

    # find top point 1: maximize distance with apex and basal center
    vd = np.linalg.norm(pts - np.tile(apex, (num_pts, 1)), axis=1)
    hd = np.linalg.norm(pts - np.tile(basal_center, (num_pts, 1)), axis=1)

    obj_func = 0.5*vd + 0.5*hd
    tp1_idx = np.argmax(obj_func)
    tp1 = pts[tp1_idx]

    # find top point 2: similarly maximize distance with newly found tp1 and apex.
    tp = np.linalg.norm(pts - np.tile(tp1, (num_pts, 1)), axis=1)
    obj =  0.5*vd  + 0.5*tp
    tp2_idx = np.argmax(obj)
    tp2 = pts[tp2_idx]

    if show:
        plt.figure()
        plt.scatter(pts[:,0], pts[:,1], c='k', s=2)
        plt.scatter(tp1[0], tp1[1], c='r', s=5)
        plt.scatter(tp2[0], tp2[1], c='r', s=5)
        plt.scatter(apex[0], apex[1], c='g', s=5)
        plt.show()

    return tp1_idx, tp2_idx

tp1_idx, tp2_idx = get_special_pts(all_pts, 0)
apex_idx = 0


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1u = v1/np.sqrt((v1 ** 2).sum(-1))[..., np.newaxis]
    v2u = v2/np.sqrt((v2 ** 2).sum(-1))[..., np.newaxis]

    projs = np.dot(v1u, v2u)

    angle = np.arccos(np.clip(projs, -1.0, 1.0))
    angle = np.arctan2(np.sin(angle), np.cos(angle)) # return smallest angle

    return np.degrees(angle)



opt_order = sort_points(all_pts, tp1_idx, 0)

sorted_pts = all_pts[opt_order]
new_tp1_idx = 0
new_tp2_idx = np.where(opt_order==tp2_idx)[0][0]

l1 = new_tp2_idx
l2 = (all_pts.shape[0]-1) - new_tp2_idx

if l1 > l2:
    no_basal_pts = np.vstack((sorted_pts[:new_tp2_idx], sorted_pts[new_tp2_idx]))
else:
    print('asds')
    no_basal_pts = np.vstack((sorted_pts[new_tp2_idx:], sorted_pts[0]))