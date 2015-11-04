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
import IDT_feature

from caffeCNN import caffe_init


sizes = np.array([[8,8], [11.4286,11.4286], [16,16], [22.8571,24], [32,34.2587]])
sizes_vid = np.array([[480,640], [340,454], [240,320], [170,227], [120,160]])


def get_tdd_flow(filenames, data_dir, use_gpu, layer, scale, gpu_id, startvid, toid):

    save_dir1 = os.path.join(data_dir, 'features', 'tdd_flow_'+layer+'_scale_'+str(scale)+'_norm_2')
    save_dir2 = os.path.join(data_dir, 'features', 'tdd_flow_'+layer+'_scale_'+str(scale)+'_norm_3')

    model_def_file = 'models/flow_'+layer+'_scale'+str(scale)+'.prototxt'
    model_file = 'temporal.caffemodel'

    # Initialize caffe net
    net = caffe_init(use_gpu, model_def_file, model_file, gpu_id)

    for i in range(startvid, toid):

        filename = filenames[i]
        filename_ = os.path.splitext(filename)[0]
        print 'Processing (%d/%d): %s' % (i+1,Nf,filename, )

        videofile = os.path.join(data_dir, 'videos', filename)

        if os.path.exists(os.path.join(save_dir2, filename_+'.mat')):
            print 'temporal TDD features exist...'
            continue

    	# iDT extraction
    	iDTF_file = os.path.join(data_dir, 'features', 'idt', filename_+'.bin')
    	
    	# TVL1 flow extraction
    	flow_file = os.path.join(data_dir, 'features', 'flow_farn', filename_)

    	# Import improved trajectories
    	IDT = IDT_feature.read_IDTF_file(iDTF_file)
        info = IDT.info
        traj = IDT.traj

        if not info.size:
            print 'IDT feature is empty...'
            continue

    	print 'Extract temporal TDD...'

    	feature = FlowCNNFeature(flow_file, net, sizes_vid[scale-1,0], sizes_vid[scale-1,1])

    	if np.amax(info[0,:]) > feature.shape[3]:
    		ind = np.where(info[0,:] <= feature.shape[3])[0]
    		info = info[:,ind]
    		traj = traj[:,ind]

    	cnn_feature1, cnn_feature2 = FeatureMapNormalization(feature)
    	idt_cnn_feature = TDD(info, traj, cnn_feature1, sizes[scale-1,0], sizes[scale-1,1], 1)
    	scipy.io.savemat(os.path.join(save_dir1, filename_+'.mat'), mdict = {'idt_cnn_feature': idt_cnn_feature})
    	idt_cnn_feature = TDD(info, traj, cnn_feature2, sizes[scale-1,0], sizes[scale-1,1], 1)
    	scipy.io.savemat(os.path.join(save_dir2, filename_+'.mat'), mdict = {'idt_cnn_feature': idt_cnn_feature})


def get_tdd_rgb(filenames, data_dir, use_gpu, layer, scale, gpu_id, startvid, toid):

    save_dir1 = os.path.join(data_dir, 'features', 'tdd_rgb_'+layer+'_scale_'+str(scale)+'_norm_2')
    save_dir2 = os.path.join(data_dir, 'features', 'tdd_rgb_'+layer+'_scale_'+str(scale)+'_norm_3')

    model_def_file = 'models/rgb_'+layer+'_scale'+str(scale)+'.prototxt'
    model_file = 'spatial.caffemodel'

    # Initialize caffe net
    net = caffe_init(use_gpu, model_def_file, model_file, gpu_id)

    for i in range(startvid, toid):

        filename = filenames[i]
        filename_ = os.path.splitext(filename)[0]
        print 'Processing (%d/%d): %s' % (i+1,Nf,filename, )

        videofile = os.path.join(data_dir, 'videos', filename)

        if os.path.exists(os.path.join(save_dir2, filename_+'.mat')):
            print 'spatial TDD features exist...'
            continue

        # iDT extraction
        iDTF_file = os.path.join(data_dir, 'features', 'idt', filename_+'.bin')
        
        # TVL1 flow extraction
        flow_file = os.path.join(data_dir, 'features', 'flow_farn', filename_)

        # Import improved trajectories
        IDT = IDT_feature.read_IDTF_file(iDTF_file)
        info = IDT.info
        traj = IDT.traj

        if not info.size:
            print 'IDT feature is empty...'
            continue

        print 'Extract spatial TDD...'

        # feature = _RGBCNNFeature(videofile, net, sizes_vid[scale-1,0], sizes_vid[scale-1,1])
        feature = RGBCNNFeature(flow_file, net, sizes_vid[scale-1,0], sizes_vid[scale-1,1])

        if np.amax(info[0,:]) > feature.shape[3]:
            ind = np.where(info[0,:] <= feature.shape[3])[0]
            info = info[:,ind]
            traj = traj[:,ind]

        cnn_feature1, cnn_feature2 = FeatureMapNormalization(feature)
        idt_cnn_feature = TDD(info, traj, cnn_feature1, sizes[scale-1,0], sizes[scale-1,1], 1)
        scipy.io.savemat(os.path.join(save_dir1, filename_+'.mat'), mdict = {'idt_cnn_feature': idt_cnn_feature})
        idt_cnn_feature = TDD(info, traj, cnn_feature2, sizes[scale-1,0], sizes[scale-1,1], 1)
        scipy.io.savemat(os.path.join(save_dir2, filename_+'.mat'), mdict = {'idt_cnn_feature': idt_cnn_feature})


# python extract_tdd.py -d /home/zhenyang/Workspace/data/UCF101/list_UCF101.txt -m rgb -l conv5 -c 3 -g 1 -s 1 -t 13320
# python extract_tdd.py -d /home/zhenyang/Workspace/data/UCF101/list_UCF101.txt -m flow -l conv5 -c 3 -g 2 -s 1 -t 13320
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FeatureExtractior')
    parser.add_argument('-d', '--dataset', dest='dataset', help='Specify dataset to process.', type=str, required=False)
    parser.add_argument('-s', '--startvid', dest='startvid', help='Specify video id start to process.', type=int, required=False)
    parser.add_argument('-t', '--tovid', dest='tovid', help='Specify video id until to process.', type=int, required=False)
    parser.add_argument('-m', '--mode', dest='mode', help='Specify mode, spatial network (rgb) or temporal network (flow).', type=str, required=True)
    parser.add_argument('-l', '--layer', dest='layer', help='Specify layer to extract features.', type=str, required=True)
    parser.add_argument('-c', '--scale', dest='scale', help='Specify scale to extract features.', type=int, required=True)
    parser.add_argument('-g', '--gpu_id', dest='gpu_id', help='Specify gpu_id to extract features.', type=int, required=True)

    args = parser.parse_args()

    if args.dataset is None:
        print 'Not specify dataset, using UCF101 by default...'
        args.dataset = '/home/zhenyang/Workspace/data/UCF101/list_UCF101.txt'

    print '***************************************'
    print '******** EXTRACT TDD FEATURES *********'
    print '***************************************'
    print 'Dataset: %s' % (args.dataset, )

    base_dir = os.path.dirname(args.dataset)
    
    filenames = []
    with open(args.dataset) as fp:
        for line in fp:
            filenames.append(line.strip())

    Nf = len(filenames)
    startvid = 0
    toid = Nf
    if args.startvid is not None and args.tovid is not None:
        startvid = max([args.startvid-1, startvid])
        toid = min([args.tovid, toid])

    if args.mode == 'flow':
        get_tdd_flow(filenames, base_dir, 1, args.layer, args.scale, args.gpu_id, startvid, toid)
    elif args.mode == 'rgb':
        get_tdd_rgb(filenames, base_dir, 1, args.layer, args.scale, args.gpu_id, startvid, toid)


    print '*********** PROCESSED ALL *************'

