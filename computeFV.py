import os, sys, collections
import numpy as np
from yael import ynumpy
import scipy.io

"""
Encodes a fisher vector.

"""


def create_fisher_vector(gmm_list, descriptor, fv_file, fv_sqrt=False, fv_l2=False):
    """
    expects a single video_descriptors object. videos_desciptors objects are defined in IDT_feature.py
    fv_file is the full path to the fisher vector that is created.

    this single video_desc contains the (trajs, hogs, hofs, mbhs) np.ndarrays
    """
    fv = np.array([])
    if descriptor.size:
        gmm, mean, pca_transform = gmm_list
        # apply the PCA to the vid_trajectory descriptor
        # each image_desc is of size (X,TRAJ_DIM). Pca_tranform is of size (TRAJ_DIM,TRAJ_DIM/2)
        descrip = descriptor.astype('float32') - mean
        if pca_transform != None:
            descrip = np.dot(descrip, pca_transform)
        
        # compute the Fisher vector, using the derivative w.r.t mu and sigma
        fv = ynumpy.fisher(gmm, descrip, include = ['mu', 'sigma'])
        
        # normalizations are done on each descriptor individually
        if fv_sqrt:
            # power-normalization
            fv = np.sign(fv) * (np.abs(fv) ** 0.5)

        if fv_l2:
            # L2 normalize
            # sum along the rows.
            norms = np.sqrt(np.sum(fv ** 2))
            # -1 allows reshape to infer the length. So it just solidifies the dimensions to (274,1)
            fv /= norms
            # handle images with 0 local descriptor (100 = far away from "normal" images)
            fv[np.isnan(fv)] = 100

    scipy.io.savemat(fv_file+'.mat', mdict={'fv':fv}, oned_as='row')
    print fv_file
    return fv
    
