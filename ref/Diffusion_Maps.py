# coding: utf-8
'''
The program calculates diffusion maps coordinates for vectors presented as rows in a file defined by a variable named <data>.
Resulting diffusion maps coordinates are stored in a variable named <coordinates>.
The number of diffusion coordinates computed is hard coded in a variable named <dim> (currently dim=5).
The top diffusion maps coordinates are typically used to embed the original data into a low dimensional Euclidean space.

 The basic steps are:
 Given N m-dimensional data points (here these are flattened sonograms) do
 1) Compute an N by N matrix that holds the pairwise distances as defined by a Gaussian kernel
 2) Normalize the kernel to be row-stochastic (sum of each row = 1)
 3) Compute the eigenvectors and eigenvalues of the normalized kernel
 4) Use the top left eigenvectors to enbed the data points. Here we use the 2st and 4rd for plotting.
 Each eigenvector is of size NX1, the ith entry corresponds to input point number i.


Last modified by Neta Rabin on 2019/02/27.
'''

import numpy as np
import sys
from numpy import linalg as LA
import matplotlib.pyplot as plt
import math
from scipy.spatial.distance import pdist, squareform

'''
epsilon_factor - a parameter that controls the width of the Gaussian kernel  
'''

'''
Compute  the width of the Gaussian kernel based on the given dataset.   
'''


# compute epsilon of (dataList)
def calcEpsilon(dataList, eps_type, epsilon_factor=4):
    dist = squareform(pdist(dataList))  # read about squareform
    if eps_type == 'maxmin':
        # option #1 - epsilon= max min(distance)
        idist = dist + np.identity(len(dist))
        eps_maxmin = np.max(np.min(idist, axis=0))
        epsilon = eps_maxmin * epsilon_factor

    elif eps_type == 'mean':
        # option #2 - epsilon= mean(distance)
        eps_mean = np.mean(dist)
        epsilon = eps_mean * epsilon_factor

    else:
        raise KeyError('eps_type should be either maxmin or mean')

    return dist, epsilon


def ker_calc(dataList, eps_type, eps):
    dist, eps = calcEpsilon(dataList, eps_type, eps)
    ker = np.exp(-(dist ** 2) / (2 * eps))
    return ker, eps


'''
Construct the NXN Gaussian kernel, normalize it and compute eigenvalues and eigenvectors.   
'''


def diffusionMapping(dataList, alpha, eps_type, eps, t,  **kwargs):
    try:
        kwargs['dim'] or kwargs['delta']
    except KeyError:
        raise KeyError('specify either dim or delta as keyword argument!')

    # compute epsilon of (dataList) eps_type can be 'mean' or 'maxmin'
    # dist is the L2 distances
    # eps_type='maxmin'#mean' #or maxmin

    ker, epsilon = ker_calc(dataList, eps_type, eps)
    v = np.sum(ker, axis=0)

    v = v ** alpha
    V_x_y = v * v[:, None]
    a = ker / V_x_y

    # calc the row sums of a, save as v1
    # in the next for-loop, divide the rows of a by v1
    sa = np.sum(a, axis=0)
    m = a / sa[:, None]

    # compute eigenvectors of (a_ij)
    vecs, eigs, _ = LA.svd(m, full_matrices=False)
    # vecs = vecs / vecs[:, 0][:, None]

    # Compute dimension
    # (for better performance you may want to combine this with an iterative way of computing eigenvalues/vectors)
    if kwargs['dim']:
        embeddim = kwargs['dim']
    elif kwargs['delta']:
        i = 1
        while eigs[i] ** t > kwargs['delta'] * eigs[1] ** t:
            i += 1
        embeddim = i

    # Compute embedding coordinates
    diffusion_coordinates = vecs[:, 1:embeddim + 1].T * (eigs[1:embeddim + 1][:, None] ** t)

    return vecs, eigs, diffusion_coordinates, dataList, epsilon


#
# data = list(np.genfromtxt("data.csv",delimiter=',')) #path to csv


'''
Plot the 2nd and 4th diffusion maps coordinates, they give nice results for this small example.
Usually you should try to plot the 2nd, 3rd,4th.. and so diffusion maps coordinates.   
'''

#
# eps_type='maxmin'#mean' #or maxmin
# alpha=1
# vecs,eigs,coordinates, dataList = diffusionMapping(data,alpha,eps_type, 1, dim=5) # dim - number of diffusion coordinates computed
# psi = np.asarray(coordinates.T)
# x = psi[:, 2] #dm cords
# y = psi[:, 3] #dm cords
# fig, ax = plt.subplots()
# labels = ['image {0}'.format(i + 1) for i in range(len(x))]
# plt.figure(); plt.scatter(vecs[:,2],vecs[:,4])
# for label, xpt, ypt in zip(labels, x, y):
#     plt.annotate(
#             "",
#             xy=(xpt, ypt), xytext=(-20, 20),
#             textcoords='offset points', ha='right', va='bottom',
#             bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
#             )
# ax.plot(x, y, 'bo')
# plt.show()
#
