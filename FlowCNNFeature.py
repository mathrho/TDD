import sys
import os
import glob

import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import cv2

from caffeCNN import caffe_init, caffe_predict

def FlowCNNFeature(vid_name, use_gpu, NUM_HEIGHT, NUM_WIDTH, model_def_file, model_file, gpu_id):

    # 2L channels
    L = 10

    # Initialize caffe net
    net = caffe_init(use_gpu,model_def_file,model_file,gpu_id)
    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    N, d1, d2, d3 = net.blobs[net.outputs[0]].data.shape

    if NUM_HEIGHT != H:
        raise Exception, 'HEIGHT is not euqual to pre-trained network config!'
    if NUM_WIDTH != W:
        raise Exception, 'WIDTH is not euqual to pre-trained network config'

    # Input video
    filelist = glob.glob(vid_name+'*_x*.jpg')
    if len(filelist) > 30*60:
        duration = 30 * 60
    else:
        duration = len(filelist)

    # get iamge mean map
    d = scipy.io.loadmat('flow_mean.mat')
    IMAGE_MEAN = d['image_mean']
    # scipy.misc.imresize only works with uint8
    # IMAGE_MEAN = imresize(IMAGE_MEAN, (NUM_HEIGHT, NUM_WIDTH), 'bicubic')
    IMAGE_MEAN = cv2.resize(IMAGE_MEAN, (NUM_WIDTH, NUM_HEIGHT), interpolation=cv2.INTER_CUBIC)

    video = np.zeros((duration, L*2, NUM_HEIGHT, NUM_WIDTH), dtype=np.float32)
    for i in range(0, duration):
        flow_x = imread( '%s_%04d.jpg' % (vid_name+'flow_x', i) )
        flow_y = imread( '%s_%04d.jpg' % (vid_name+'flow_y',,i) )

        # RGB -> BGR, not need here
        # resize scipy.misc.imresize only works with uint8
        flow_x = cv2.resize(flow_x, (NUM_WIDTH, NUM_HEIGHT), interpolation=cv2.INTER_LINEAR)
        flow_y = cv2.resize(flow_y, (NUM_WIDTH, NUM_HEIGHT), interpolation=cv2.INTER_LINEAR)
        # mean subtraction
        flow_x = flow_x - IMAGE_MEAN
        flow_y = flow_y - IMAGE_MEAN
        #
        video(i,0,:,:) = flow_x
        video(i,1,:,:) = flow_y

    for i in range(0, L-1)
        tmp = concatenate((video[1:duration,i*2:(i+1)*2,:,:],video[-1,i*2:(i+1)*2,:,:]), axis=0)
        video[;,(i+1)*2:(i+2)*2,:,:] = tmp

    # Computing convoltuional maps
    # Keep in mind that width is the fastest dimension and channels are BGR (in Matlab)
    # however, Matlab (W,H,C,N) -> Python (N,C,H,W)
    batch_size = N # batch_size = 40
    batch_images = np.zeros((batch_size, L*2, NUM_HEIGHT, NUM_WIDTH), dtype=np.float32)

    FlowFeature = np.zeros((duration, d1, d2, d3), dtype=np.float32)
    for j in range(0, duration, batch_size):
        batch_range = range(j, min(j+batch_size, duration))
        batch_images[0:len(batch_range),:,:,:] = video[batch_range,:,:,:]

        # Predict CNN feature
        feature = caffe_predict(batch_images, net)
        FCNNFeature[batch_range,:,:,:] = feature[0:len(batch_range),:,:,:]

    # tranpose to matlab format (N,C,H,W) -> (H,W,C,N) not (W,H,C,N)
    FCNNFeature = np.transpose(FCNNFeature, (2,3,1,0))

    return FCNNFeature

