import sys
import os.path

import caffe

def caffe_init(use_gpu, model_def_file, model_file, gpu_id):
    """
    Initilize pycaffe wrapper
    """

    
    if use_gpu:
        print 'Using GPU Mode'
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
    else:
        print 'Using CPU Mode'
        caffe.set_mode_cpu()

    # By default use imagenet_deploy
    # model_def_file = 'models/UCF_CNN_M_2048_deploy.prototxt'
    # By default use caffe reference model
    # model_file = 'models/1_vgg_m_fine_tuning_rgb_iter_20000.caffemodel'
    if os.path.isfile(model_file):
        # NOTE: you'll have to get the pre-trained ILSVRC network
        print 'You need a network model file'

    if os.path.isfile(model_def_file):
        # NOTE: you'll have to get network definition
        print 'You need the network prototxt definition'

    # run with phase test (so that dropout isn't applied)
    net = caffe.Net(model_def_file, model_file, caffe.TEST)
    #caffe.set_phase_test()
    print 'Done with init, Done with set_phase_test'

    return net


def caffe_predict(in_data, net):
    """
    Get the features for a batch of data using network

    Inputs:
    in_data: data batch
    """

    out = net.forward(**{net.inputs[0]: in_data})
    #features = out[net.outputs[0]].squeeze(axis=(2,3))
    features = out[net.outputs[0]]
    return features

