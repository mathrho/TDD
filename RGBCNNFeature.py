import sys
import os

import numpy as np
from scipy.misc import imresize
import scipy.io
import cv2

from caffeCNN import caffe_init, caffe_predict

def RGBCNNFeature(vid_name, use_gpu, NUM_HEIGHT, NUM_WIDTH, model_def_file, model_file, gpu_id):

    # Initialize caffe net
    net = caffe_init(use_gpu,model_def_file,model_file,gpu_id)
    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    N, d1, d2, d3 = net.blobs[net.outputs[0]].data.shape

    if NUM_HEIGHT != H:
        raise Exception, 'HEIGHT is not euqual to pre-trained network config!'
    if NUM_WIDTH != W:
        raise Exception, 'WIDTH is not euqual to pre-trained network config'

    # Input video
    vidCap = cv2.VideoCapture(vid_name)
    if not vidCap.isOpened():  # check if we succeeded
        print 'Could not initialize capturing..%s' % (vid_name, )
        return

    numFrame = int(vidCap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if numFrame > 30*60:
        duration = 30 * 60
    else:
        duration = numFrame
    #
    d = scipy.io.loadmat('VGG_mean.mat')
    IMAGE_MEAN = d['image_mean']
    # scipy.misc.imresize only works with uint8
    # IMAGE_MEAN = imresize(IMAGE_MEAN, (NUM_HEIGHT, NUM_WIDTH), 'bicubic')
    IMAGE_MEAN = cv2.resize(IMAGE_MEAN, (NUM_WIDTH, NUM_HEIGHT), interpolation=cv2.INTER_CUBIC)

    video = np.zeros((duration, 3, NUM_HEIGHT, NUM_WIDTH), dtype=np.float32)
    for i in range(0, duration):
        flag, frame = vidCap.read()
        if flag:
            # The frame is ready and already captured
            #if len(frame.shape) == 2:
            #    frame = np.tile(frame[:,:,np.newaxis], (1,1,3))

            # OpenCV BGR -> RGB ?? (caffe uses BGR)
            # frame = frame[:,:,(2,1,0)]
            # resize: scipy.misc.imresize only works with uint8
            # frame = imresize(frame, (NUM_HEIGHT, NUM_WIDTH), 'bilinear')
            frame = cv2.resize(frame, (NUM_WIDTH, NUM_HEIGHT), interpolation=cv2.INTER_LINEAR)
            import pdb; pdb.set_trace()
            # mean subtraction
            frame = frame - IMAGE_MEAN
            # get channel in correct dimension (H,W,C) -> (C,H,W)
            frame = np.transpose(frame, (2,0,1))
            #
            video[i,:,:,:] = frame
        else:
            # The next frame is not ready, so we try to read it again
            print 'frame is not ready'

    # Computing convoltuional maps
    # Keep in mind that width is the fastest dimension and channels are BGR (in Matlab)
    # however, Matlab (W,H,C,N) -> Python (N,C,H,W)
    batch_size = N # batch_size = 40
    batch_images = np.zeros((batch_size, 3, NUM_HEIGHT, NUM_WIDTH), dtype=np.float32)

    FCNNFeature = np.zeros((duration, d1, d2, d3), dtype=np.float32)
    for j in range(0, duration, batch_size):
        batch_range = range(j, min(j+batch_size, duration))
        batch_images[0:len(batch_range),:,:,:] = video[batch_range,:,:,:]

        # Predict CNN feature
        feature = caffe_predict(batch_images, net)
        FCNNFeature[batch_range,:,:,:] = feature[0:len(batch_range),:,:,:]

    # tranpose to matlab format (N,C,H,W) -> (H,W,C,N) not (W,H,C,N)
    FCNNFeature = np.transpose(FCNNFeature, (2,3,1,0))

    return FCNNFeature

