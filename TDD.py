import sys
import os.path

import numpy as np

def TDD(inf,tra,cnn_feature,scale_x,scale_y,num_cell):

	if inf is not None:
		ind = np.where(inf[6,:] == 1)[0]
		inf = inf[:,ind]
		tra = tra[:,ind]

	if inf is not None:
		NUM_DIM = cnn_feature.shape[2]
		NUM_DES = inf.shape[1]
		TRA_LEN = tra.shape[0]/2

		num_fea = TRA_LEN / num_cell

		pos = np.reshape(tra, (2,-1), order='F') - 1
		pos = np.divide(pos, [[scale_x],[scale_y]]) + 1
		#pos = np.around(pos) round half-number round even number, 0.5->0
		pos = np.floor(np.abs(pos) + 0.5) * np.sign(pos) 
		pos = np.fmax(pos, [[1],[1]])
		pos = np.fmin(pos, [[cnn_feature.shape[1]],[cnn_feature.shape[0]]])
		pos = np.reshape(pos, (TRA_LEN*2,-1))

		cnn_feature = np.transpose(cnn_feature, (0,1,3,2))
		offset = np.arange(TRA_LEN-1,-1,-1)
		size_mat = [cnn_feature.shape[0],cnn_feature.shape[1],cnn_feature.shape[2]]
		cnn_feature = np.reshape(cnn_feature, (-1,NUM_DIM))

		cur_x = pos[range(0,TRA_LEN*2,2),:]
		cur_y = pos[range(1,TRA_LEN*2,2),:]
		cur_t= np.subtract(inf[0,:], np.transpose(offset[np.newaxis,:]))

		import pdb; pdb.set_trace()
		tmp = cnn_feature[np.ravel_multi_index([cur_y-1,cur_x-1,cur_t-1],size_mat,order='F'), :]
		tmp = np.transpose(tmp)
		tmp = np.reshape(tmp, (NUM_DIM,num_fea,-1))
		feature = np.reshape(np.sum(tmp,axis=1), (-1,NUM_DES))

	else:

		feature = []

	return feature

