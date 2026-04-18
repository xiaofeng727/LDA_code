# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:00:57 2022

@author: Chi Ding
"""


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
import copy
import math
import scipy.io as io
#%% define LDA net

#%% dataset
class RandomDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.len = length
        
    def __getitem__(self, index):
        return torch.Tensor(self.data[index,:]).float()
    
    def __len__(self):
        return self.len
#%% helper functions
def load_Phi(cs_ratio):
    Phi_data_Name = 'phi/'+'phi_0_%d_1089.mat' % cs_ratio
    Phi_data = io.loadmat(Phi_data_Name)

    # used for compressive sensing sampling
    Phi_input = Phi_data['phi']
    
    return Phi_input
def compute_initialization_matrix(cs_ratio):
    
    Phi_input = load_Phi(cs_ratio)
    
    Training_data_Name = 'data/Training_Data_Img91.mat'
    Training_data = io.loadmat(Training_data_Name)

    # labels are 88912 original images
    Training_labels = Training_data['labels']

    # Computing Initialization Matrix
    X = Training_labels.T

    # c-s sampling x into y
    Y = Phi_input @ X

    # Compute Q_init
    Qinit = X @ Y.T @ np.linalg.inv(Y @ Y.T)
    del X, Y
    
    return Qinit


def rgb2ycbcr(rgb):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = rgb.shape
    if len(shape) == 3:
        rgb = rgb.reshape((shape[0] * shape[1], 3))
    ycbcr = np.dot(rgb, m.transpose() / 255.)
    ycbcr[:,0] += 16.
    ycbcr[:,1:] += 128.
    return ycbcr.reshape(shape)

# ITU-R BT.601
# https://en.wikipedia.org/wiki/YCbCr
# YUV -> RGB
def ycbcr2rgb(ycbcr):
    m = np.array([[ 65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112],
                  [ 112, -93.786, -18.214]])
    shape = ycbcr.shape
    if len(shape) == 3:
        ycbcr = ycbcr.reshape((shape[0] * shape[1], 3))
    rgb = copy.deepcopy(ycbcr)
    rgb[:,0] -= 16.
    rgb[:,1:] -= 128.
    rgb = np.dot(rgb, np.linalg.inv(m.transpose()) * 255.)
    return rgb.clip(0, 255).reshape(shape)


def imread_CS_py(Iorg):
    block_size = 33
    [row, col] = Iorg.shape
    row_pad = block_size-np.mod(row,block_size)
    col_pad = block_size-np.mod(col,block_size)
    Ipad = np.concatenate((Iorg, np.zeros([row, col_pad])), axis=1)
    Ipad = np.concatenate((Ipad, np.zeros([row_pad, col+col_pad])), axis=0)
    [row_new, col_new] = Ipad.shape

    return [Iorg, row, col, Ipad, row_new, col_new]


def img2col_py(Ipad, block_size):
    [row, col] = Ipad.shape
    row_block = row/block_size
    col_block = col/block_size
    block_num = int(row_block*col_block)
    img_col = np.zeros([block_size**2, block_num])
    count = 0
    for x in range(0, row-block_size+1, block_size):
        for y in range(0, col-block_size+1, block_size):
            img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].reshape([-1])
            # img_col[:, count] = Ipad[x:x+block_size, y:y+block_size].transpose().reshape([-1])
            count = count + 1
    return img_col


def col2im_CS_py(X_col, row, col, row_new, col_new):
    block_size = 33
    X0_rec = np.zeros([row_new, col_new])
    count = 0
    for x in range(0, row_new-block_size+1, block_size):
        for y in range(0, col_new-block_size+1, block_size):
            X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size])
            # X0_rec[x:x+block_size, y:y+block_size] = X_col[:, count].reshape([block_size, block_size]).transpose()
            count = count + 1
    X_rec = X0_rec[:row, :col]
    return X_rec


def psnr(img1, img2):
    # 确保输入为浮点型张量
    img1 = img1.type(torch.float32)  # 或 img1.float()
    img2 = img2.type(torch.float32)  # 或 img2.float()

    # 计算均方误差 (MSE)
    mse = torch.mean((img1 - img2) ** 2)

    if mse == 0:
        return 100  # 完美匹配时返回最大值

    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse.item()))




def ssim(img1, img2, window_size=11, sigma=1.5, data_range=1.0):
    """
    支持四维输入的SSIM计算 (Batch, Channel, Height, Width)
    """
    if img1.dim() != 4 or img2.dim() != 4:
        raise ValueError(
            f"Input tensors must be 4D [B, C, H, W]. "
            f"Got img1: {img1.shape}, img2: {img2.shape}"
        )
    # 确保输入为四维张量
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    assert img1.dim() == 4 and img2.dim() == 4, "Input must be 4D [B, C, H, W]"

    img1 = img1.float()
    img2 = img2.float()

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # 生成高斯核（支持多通道）
    coords = torch.arange(window_size).float() - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss /= gauss.sum()
    gauss = gauss.view(1, 1, window_size, 1)  # [1, 1, W, 1]
    gauss = gauss.repeat(img1.shape[1], 1, 1, 1)  # [C, 1, W, 1]
    gauss = gauss.to(img1.device)

    def gaussian_conv(x):
        # 分组卷积（每组一个通道）
        x = F.conv2d(x, gauss, padding=window_size//2, groups=x.shape[1])
        x = F.conv2d(x, gauss.transpose(2,3), padding=window_size//2, groups=x.shape[1])
        return x

    mu1 = gaussian_conv(img1)
    mu2 = gaussian_conv(img2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_conv(img1 ** 2) - mu1_sq
    sigma2_sq = gaussian_conv(img2 ** 2) - mu2_sq
    sigma12 = gaussian_conv(img1 * img2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
