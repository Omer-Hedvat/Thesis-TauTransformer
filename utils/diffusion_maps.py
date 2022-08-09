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
from numpy import linalg as LA
from scipy.spatial.distance import pdist, squareform
import pandas as pd

'''
epsilon_factor - a parameter that controls the width of the Gaussian kernel  
'''

'''
Compute  the width of the Gaussian kernel based on the given dataset.   
'''


# compute epsilon of (dataList)
def calc_epsilon(data_list, eps_type, epsilon_factor=4):
    distance_np = squareform(pdist(data_list))  # read about squareform
    if eps_type == 'maxmin':
        # option #1 - epsilon= max min(distance)
        idist = distance_np + np.identity(len(distance_np))
        eps_maxmin = np.max(np.min(idist, axis=0))
        eps = eps_maxmin * epsilon_factor

    elif eps_type == 'mean':
        # option #2 - epsilon= mean(distance)
        eps_mean = np.mean(distance_np)
        eps = eps_mean * epsilon_factor

    else:
        raise KeyError('eps_type should be either maxmin or mean')

    return distance_np, eps


def kernel_calc(datalist, eps_type, epsilon_factor):
    distance_np, eps = calc_epsilon(datalist, eps_type, epsilon_factor)
    kernel = np.exp(-(distance_np ** 2) / (2 * eps))
    return kernel, eps


'''
Construct the NXN Gaussian kernel, normalize it and compute eigenvalues and eigenvectors.   
'''


def diffusion_mapping(data_list, alpha, eps_type, epsilon_factor, **kwargs):
    """

    :param data_list:
    :param alpha:
    :param eps_type:
    :param epsilon_factor:
    :param t:
    :param kwargs:
    :return:
    """
    assert 'dim' in kwargs.keys()

    # compute epsilon of (data_list) eps_type can be 'mean' or 'maxmin'
    # dist is the L2 distances
    # eps_type='maxmin'#mean' #or maxmin

    kernel, epsilon = kernel_calc(data_list, eps_type, epsilon_factor)
    v = np.sum(kernel, axis=0)

    # 1st normalization - treats uneven density
    v = v ** alpha
    V_x_y = v * v[:, None]
    a = kernel / V_x_y

    # calc the row sums of a, save as v1
    # in the next for-loop, divide the rows of a by v1
    # 2nd normalizaton - sum of every row equals to 1
    sa = np.sum(a, axis=0)
    m = a / sa[:, None]

    # compute eigenvectors of (a_ij)
    singular_vectors, singular_values, _ = LA.svd(m, full_matrices=False)

    # Compute embedding coordinates
    diffusion_coordinates = singular_vectors[:, 1:kwargs['dim'] + 1].T * (singular_values[1:kwargs['dim'] + 1][:, None])
    ranking = singular_vectors[:, :1]

    return {'coordinates': diffusion_coordinates, 'ranking': ranking}


def main():

    # eps_type ='maxmin' # mean or maxmin
    # alpha = 1
    # vecs,eigs,coordinates, dataList = diffusion_mapping(data,alpha,eps_type, 1, dim=5) # dim - number of diffusion coordinates computed
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

    df_glass = pd.read_csv('data/glass.csv')

    eps_type = 'maxmin'  # mean' #or maxmin
    alpha = 1
    singular_vectors, singular_values, diffusion_coordinates, data_list, epsilon, ranking = diffusion_mapping(df_glass, alpha, eps_type, 8, 1, dim=2)
    pass


if __name__ == '__main__':
    main()
