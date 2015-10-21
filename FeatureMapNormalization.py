import sys
import os

import numpy as np


def FeatureMapNormalization(cnn_feature):

	r, c, f, t = cnn_feature.shape

	#(H,W,C,N) -> (H,W,N,C), matlab first index changing fastest -> python last axis changing fastest
	cnn_feature1 = np.transpose(cnn_feature, (0,1,3,2))
	cnn_feature1 = np.reshape(cnn_feature1, (r*c*t,f), order='F')
	
	max_cnn_feature1 = np.amax(cnn_feature1, axis=0) + np.finfo(np.float32).eps
	cnn_feature1 = np.divide(cnn_feature1, max_cnn_feature1) # np.tile(max_cnn_feature1, (r*c*t,1))

	cnn_feature1 = np.reshape(cnn_feature1, (r,c,t,f), order='F')
	cnn_feature1 = np.transpose(cnn_feature1, (0,1,3,2))

	max_cnn_feature = np.amax(cnn_feature, axis=2) + np.finfo(np.float32).eps
	cnn_feature2 = np.divide(cnn_feature, max_cnn_feature[:,:,np.newaxis,:]) # np.tile(max_cnn_feature[:,:,np.newaxis,:], (1,1,f,1))

	return cnn_feature1, cnn_feature2
