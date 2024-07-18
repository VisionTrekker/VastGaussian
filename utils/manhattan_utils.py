# -*- coding: utf-8 -*-
#        Data: 2024-06-21 15:59
#     Project: VastGaussian
#   File Name: manhattan_utils.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

import math
import numpy as np

def create_man_rans(position, rotation):
    # create manhattan transformation matrix for threejs
    # The angle is reversed because the counterclockwise direction is defined as negative in three.js
    # 根据传入的旋转角度，分别计算分别绕 x、y 和 z 轴旋转的旋转矩阵（由于 three.js 中逆时针旋转方向定义为负，因此这里的旋转角度取反）
    rot_x = np.array([[1, 0, 0],
                      [0, math.cos(np.deg2rad(-rotation[0])), -math.sin(np.deg2rad(-rotation[0]))],
                      [0, math.sin(np.deg2rad(-rotation[0])),  math.cos(np.deg2rad(-rotation[0]))]])
    rot_y = np.array([[ math.cos(np.deg2rad(-rotation[1])), 0, math.sin(np.deg2rad(-rotation[1]))],
                      [0, 1, 0],
                      [-math.sin(np.deg2rad(-rotation[1])), 0, math.cos(np.deg2rad(-rotation[1]))]])
    rot_z = np.array([[math.cos(np.deg2rad(-rotation[2])), -math.sin(np.deg2rad(-rotation[2])), 0],
                      [math.sin(np.deg2rad(-rotation[2])),  math.cos(np.deg2rad(-rotation[2])), 0],
                      [0, 0, 1]])

    rot = rot_z @ rot_y @ rot_x     # 最终的旋转矩阵
    man_trans = np.zeros((4, 4))
    man_trans[:3, :3] = rot.transpose()
    man_trans[:3, -1] = np.array(position).transpose()
    man_trans[3, 3] = 1

    return man_trans


def get_man_trans(lp):
    lp.pos = [float(pos) for pos in lp.pos.split(" ")]
    lp.rot = [float(rot) for rot in lp.rot.split(" ")]

    man_trans = None
    if lp.manhattan and lp.platform == "tj":  # threejs, 则pos和rot的参数个数均为三个
        man_trans = create_man_rans(lp.pos, lp.rot) # 经过曼哈顿对齐后的点云坐标 相对于 初始点云坐标的变换矩阵 T_初始_曼哈顿对齐后
        lp.man_trans = man_trans
    elif lp.manhattan and lp.platform == "cc":  # cloudcompare，则rot为旋转矩阵
        rot = np.array(lp.rot).reshape([3, 3])
        man_trans = np.zeros((4, 4))
        man_trans[:3, :3] = rot
        man_trans[:3, -1] = np.array(lp.pos)
        man_trans[3, 3] = 1

    return man_trans