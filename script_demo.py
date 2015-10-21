import sys
import os.path
import argparse
import json

import numpy as np
import scipy.io

from RGBCNNFeature import RGBCNNFeature
from FlowCNNFeature import FlowCNNFeature
from FeatureMapNormalization import FeatureMapNormalization
from TDD import TDD

sizes = np.array([[8,8], [11.4286,11.4286], [16,16], [22.8571,24], [32,34.2587]])
sizes_vid = np.array([[480,640], [340,454], [240,320], [170,227], [120,160]])

def main(options):
	#import pdb; pdb.set_trace()

	# iDT extraction
	print 'Extract improved trajectories...'
	#system(['./DenseTrackStab -f ',vid_name,' -o ',vid_name(1:end-4),'.bin']);

	# TVL1 flow extraction
	print 'Extract TVL1 optical flow field...'
	#mkdir test/
	#system(['./denseFlow -f ',vid_name,' -x test/flow_x -y test/flow_y -b 20 -t 1 -d 3']);
	#system(['./denseFlow_gpu -d 1 -f ',vid_name,' -x test/flow_x -y test/flow_y -b 20 -t 1 -d 3']);

	# Import improved trajectories
	#IDT = import_idt('test.bin',15);
	#info = IDT.info;
	#tra = IDT.tra;

	# Spatial TDD
	print 'Extract spatial TDD...'

	scale = 3
	layer = 'conv5'
	gpu_id = 1

	model_def_file = 'models/rgb_'+layer+'_scale'+str(scale)+'.prototxt'
	model_file = 'spatial.caffemodel'

	feature_conv = RGBCNNFeature(options['videofile'], 1, sizes_vid[scale-1,0], sizes_vid[scale-1,1], model_def_file, model_file, gpu_id)
	feature_conv_normalize_1, feature_conv_normalize_2 = FeatureMapNormalization(feature_conv);

	tdd_feature_spatial_1 = TDD(info, tra, feature_conv_normalize_1, sizes[scale-1,0], sizes[scale-1,1], 1)
	tdd_feature_spatial_2 = TDD(info, tra, feature_conv_normalize_2, sizes[scale-1,0], sizes[scale-1,2], 1)

	scipy.io.savemat(os.path.join('./', 'rgbCNNFeature_py.mat'), mdict = {'tdd_feature_spatial_1': tdd_feature_spatial_1, 'tdd_feature_spatial_2': tdd_feature_spatial_2})


	# Temporal TDD
	# caffe.reset_all()

	print 'Extract temporal TDD...'

	scale = 3
	layer = 'conv5'
	gpu_id = 1

	model_def_file = 'models/flow_'+layer+'_scale'+str(scale)+'.prototxt'
	model_file = 'temporal.caffemodel'

	feature_conv = FlowCNNFeature('test/', 1, sizes_vid[scale-1,0], sizes_vid[scale-1,1], model_def_file, model_file, gpu_id)
	feature_conv_normalize_1, feature_conv_normalize_2 = FeatureMapNormalization(feature_conv);

	tdd_feature_temporal_1 = TDD(info, tra, feature_conv_normalize_1, sizes[scale-1,0], sizes[scale-1,1], 1)
	tdd_feature_temporal_2 = TDD(info, tra, feature_conv_normalize_2, sizes[scale-1,0], sizes[scale-1,2], 1)

	scipy.io.savemat(os.path.join('./', 'flowCNNFeature_py.mat'), mdict = {'tdd_feature_temporal_1': tdd_feature_temporal_1, 'tdd_feature_temporal_2': tdd_feature_temporal_2})


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	#
	parser.add_argument('-v', '--videofile', dest='videofile', default='test.avi', help='video: filename')


	args = parser.parse_args()
	options = vars(args) # convert to ordinary dict
	print 'parsed option parameters:'
	print json.dumps(options, indent = 2)
	main(options)

