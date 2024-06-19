import numpy as np
import math
import pandas as pd
from matplotlib import cm 
from matplotlib import pyplot as plt
import scvelo as scv
from torch.utils.data import Dataset, DataLoader

'''
Input: xy - [n x 2] array for the coordinate of points.
       z  - [n x d] array for the features attached to each point.
Output: a X by Y by d tensor A that contains the (avg.) feature value for the points as a grid. Might contain NaN if there is no value.
        (x_lb, x_rb, xlen) and (y_lb, y_rb, ylen) as info for x, y axis
'''
def discretize(xy, z, cell_size=32):
    if isinstance(z[0], float):
        d = 1
    else:
        d = len(z[0])

    # get grid boundaries
    x_max, x_min = np.max(xy[:, 0]), np.min(xy[:, 0])
    y_max, y_min = np.max(xy[:, 1]), np.min(xy[:, 1])

    x_lb = x_min - 0.5 * cell_size
    xlen = math.ceil((x_max - x_lb) / cell_size)
    x_rb = x_lb + xlen * cell_size

    y_lb = y_min - 0.5 * cell_size
    ylen = math.ceil((y_max - y_lb) / cell_size)
    y_rb = y_lb + ylen * cell_size

    # A: output matrix. A[i][j] is the average of z values for points that will be in this cell
    # count: count how many points will be in each cell
    A = np.nan * np.empty((xlen, ylen, d))
    count = np.zeros((xlen, ylen))
    xy_new = []

    for [px, py], pz in zip(xy, z):
        # get point location in grid
        x = int((px - x_lb) // cell_size)
        y = int((py - y_lb) // cell_size)
        count[x, y] += 1
        xy_new.append([x, y])
        
        # no previous points
        if count[x, y] == 1:
            A[x, y] = pz
        else:
            A[x, y] += (pz - A[x, y]) / count[x, y]
    print(xlen, ylen)
    return A, xy_new, (x_lb, x_rb, xlen), (y_lb, y_rb, ylen)

'''
Input: A - [X x Y x d] tensor for the input values. Might contain NaN
       xy - the coordinate of non-zero points
Output: Dx, Dy - two [X x Y x d] tensors for the numerically estimated gradient. Might contain NaN

threshold: the farthest distance to consider. Sometimes the data point might be two sparse, causing the failure to predict any gradients. (TO BE IMPLEMENTED)

Assume that an entry in input A is either all NaN, or all not NaN.
'''
def numeric_gradient(xy_new, A, threshold=1):
    Dx = np.nan * np.empty_like(A)
    Dy = np.nan * np.empty_like(A)
    xlen, ylen, d = A.shape
    xgrad, ygrad = [], []

    for q in xy_new:
        i, j = int(q[0]), int(q[1])
        # if there is no cell, skip
        if np.isnan(A[i][j][0]):
            continue

        # for Dx
        # avail: how many samples are available (left and right)
        # diff: accumulated differences: e.g. (A[i][j] - A[i][j-1]) + (A[i][j+1] - A[i][j])
        avail = 0
        diff = np.zeros(d)
        if j != 0 and (not np.isnan(A[i][j-1][0])):
            avail += 1
            diff += A[i][j] - A[i][j-1]
        if j != ylen - 1 and (not np.isnan(A[i][j+1][0])):
            avail += 1
            diff += A[i][j+1] - A[i][j]
        if avail != 0:
            Dx[i][j] = diff / avail

        # for Dy, reset the local var
        avail = 0
        diff = np.zeros(d)
        if i != 0 and (not np.isnan(A[i-1][j][0])):
            avail += 1
            diff += A[i][j] - A[i-1][j]
        if i != xlen - 1 and (not np.isnan(A[i+1][j][0])):
            avail += 1
            diff += A[i+1][j] - A[i][j]
        if avail != 0:
            Dy[i][j] = diff / avail

        xgrad.append(Dx[i][j])
        ygrad.append(Dy[i][j])
    return np.array(xgrad), np.array(ygrad), Dx, Dy

class SpatialDataset(Dataset):
    def __init__(self, xy, s, u, transform=None):
        assert len(xy) == len(s) and len(s) == len(u)
        self.xy = xy
        self.s = s
        self.u = u
        self.transform = transform

    def __len__(self):
        return len(self.xy)
    
    def __getitem__(self, idx):
        sample = self.xy[idx], self.s[idx], self.u[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample

class SpatialGradientDataset(Dataset):
    def __init__(self, xy, latent, grad, transform=None):
        assert (len(xy) == len(latent)) and len(latent) == len(grad)
        self.xy = xy
        self.latent = latent
        self.grad = grad
        self.transform = transform

    def __len__(self):
        return len(self.xy)
    
    def __getitem__(self, idx):
        sample = self.xy[idx], self.latent[idx], self.grad[idx]

        if self.transform:
            sample = self.transform(sample)
        return sample