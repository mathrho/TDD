import sys
import os.path
import argparse
import json

import numpy as np
import scipy.io

from RGBCNNFeature import RGBCNNFeature

sizes = np.array([[8,8], [11.4286,11.4286], [16,16], [22.8571,24], [32,34.2587]])
sizes_vid = np.array([[480,640], [340,454], [240,320], [170,227], [120,160]])

def main(options):
	#import pdb; pdb.set_trace()

	# Spatial TDD
	print 'Extract spatial TDD...'

	scale = 3
	layer = 'conv5'
	gpu_id = 1

	model_def_file = 'models/rgb_'+layer+'_scale'+str(scale)+'.prototxt'
	model_file = 'spatial.caffemodel'

	feature_conv = RGBCNNFeature(options['videofile'], 1, sizes_vid[scale,0], sizes_vid[scale,1], model_def_file, model_file, gpu_id)

	scipy.io.savemat(os.path.join('./', 'FCNNFeature_py.mat'), mdict = {'FCNNFeature': feature_conv})


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	#
	parser.add_argument('-v', '--videofile', dest='videofile', default='test.avi', help='video: filename')


	args = parser.parse_args()
	options = vars(args) # convert to ordinary dict
	print 'parsed option parameters:'
	print json.dumps(options, indent = 2)
	main(options)

