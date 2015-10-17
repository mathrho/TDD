import sys
import os

import numpy as np
from scipy.misc import imresize
import scipy.io
import cv2

from caffeCNN import caffe_init, caffe_predict

def RGBCNNFeature(vid_name, use_gpu, NUM_HEIGHT, NUM_WIDTH, model_def_file, model_file, gpu_id):

    # Input video
    vidCap = cv2.VideoCapture(vid_name)
    if(!vidCap.isOpened()):  # check if we succeeded
        print 'Could not initialize capturing..%s' % (vid_name, )
        return

    numFrame = int(vidCap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if length > 30*60:
        duration = 30 * 60
    else:
        duration = numFrame
    #
    d = scipy.io.loadmat('VGG_mean.mat')
    IMAGE_MEAN = d['image_mean'];
    IMAGE_MEAN = imresize(IMAGE_MEAN, (NUM_HEIGHT, NUM_WIDTH));

    video = np.zeros((NUM_HEIGHT, NUM_WIDTH, 3, duration), dtype=np.float32)
    for i in range(0, duration):
        flag, frame = vidCap.read()
        if flag:
            # The frame is ready and already captured
            if len(frame.shape) == 2:
                frame = np.tile(frame[:,:,np.newaxis], (1,1,3))

            # OpenCV BGR -> RGB (caffe uses BGR)
            # frame = frame[:,:,(2,1,0)]
            # resize
            frame = imresize(frame, (NUM_HEIGHT, NUM_WIDTH), 'bilinear')
            # mean subtraction
            frame = frame - IMAGE_MEAN
            # get channel in correct dimension
            # frame = np.transpose(frame, (1,0,2))
            video[:,:,:,i] = frame
        else:
            # The next frame is not ready, so we try to read it again
            print 'frame is not ready'

    # Initialize caffe net
    net = caffe_init(use_gpu,model_def_file,model_file,gpu_id)
    N, C, H, W = net.blobs[net.inputs[0]].data.shape
    N, C, F,  = net.blobs[net.outputs[0]].data.shape

    # Computing convoltuional maps
    # BGR -> RGB
    # video = video[:,:,[3,2,1],:]
    # (H,W) -> (W,H)
    video = np.transpose(video, (1,0,2,3))

    batch_size = N # batch_size = 40
    FCNNFeature = np.zeros((, , , duration), dtype=np.float32)
    batch_images = np.zeros((NUM_WIDTH, NUM_HEIGHT, 3, batch_size), dtype=np.float32)
    for j in range(0, duration, batch_size):
        batch_range = range(j, min(j+batch_size, duration))
        batch_images[:,:,:,0:len(batch_range)] = video[:,:,:,batch_range]

        feature = caffe_predict(batch_images, net)
        feature = np.transpose(feature, (1,0,2,3))

        FCNNFeature[:,:,:,batch_range] = feature[:,:,:,0:len(batch_range)]


# Keep in mind that width is the fastest dimension and channels are BGR

% Computing convoltuional maps
d = load('VGG_mean');
IMAGE_MEAN = d.image_mean;
IMAGE_MEAN = imresize(IMAGE_MEAN,[NUM_HEIGHT,NUM_WIDTH]);
video = video(:,:,[3,2,1],:);
video = bsxfun(@minus,video,IMAGE_MEAN);
video = permute(video,[2,1,3,4]);

batch_size = 40;
num_images = size(video,4);
num_batches = ceil(num_images/batch_size);
FCNNFeature = [];

images = zeros(NUM_WIDTH, NUM_HEIGHT, 3, batch_size, 'single');
for bb = 1 : num_batches
    range = 1 + batch_size*(bb-1): min(num_images,batch_size*bb);
    tmp = video(:,:,:,range);
    images(:,:,:,1:size(tmp,4)) = tmp;
    
    feature = net.forward({images}); %feature = caffe('forward',{images});
    feature = permute(feature{1},[2,1,3,4]);
    if isempty(FCNNFeature)
        FCNNFeature = zeros(size(feature,1), size(feature,2), size(feature,3), num_images, 'single');
    end
    FCNNFeature(:,:,:,range) = feature(:,:,:,mod(range-1,batch_size)+1);
end














