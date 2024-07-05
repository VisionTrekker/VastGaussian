#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    # 将世界坐标系（World Coordinate System，WCS）中的点投影到相机坐标系（View Coordinate System，VCS）中
    Rt = np.zeros((4, 4))   # W2C
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3] # 相机在世界坐标系中的坐标位置
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W) # W2C的变换矩阵
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    这个函数的目的是计算3D场景中的投影矩阵，以便在图形学或其他应用中使用
        znear：视远的 近裁剪面 到 视图摄像机的距离
        zfar：视远的 远裁剪面 到 视图摄像机的距离
        fovX：水平视野角度（以度为单位）
        fovY：垂直视野角度（以度为单位）
        函数首先计算tan(fov/2)，然后计算视远裁剪面的四个边距（top、bottom、right、left）
        接下来，函数创建一个4x4的零矩阵P。然后，根据输入参数计算P的各个元素。最后，函数返回P，即投影矩阵
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))