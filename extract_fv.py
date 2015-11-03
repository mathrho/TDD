import sys
import os.path
import argparse
import os, subprocess, ThreadPool
import numpy as np
import scipy.io
import computeFV


# This is is the function that each worker will compute.
def processVideo(vid,TDD_DIR,FV_DIR,gmm_list):
    """
    Extracts the IDTFs, constructs a Fisher Vector, and saves the Fisher Vector at FV_DIR
    output_file: the full path to the newly constructed fisher vector.
    gmm_list: a list of gmms
    """
    input_file = os.path.join(TDD_DIR, vid.split('.')[0]+'.mat')
    output_file = os.path.join(FV_DIR, vid.split('.')[0]+'.fv')

    if not os.path.exists(input_file):
        print '%s TDD Feature does not exist!' % vid
        return False

    if os.path.exists(output_file+'.mat'):
        print '%s Fisher Vector exists, skip!' % vid
        return False

    data = scipy.io.loadmat(input_file)
    tdd = data['idt_cnn_feature'].T

    # why sqrt? just like root sift! power: 0.5
    tdd = np.sqrt(tdd)
    
    computeFV.create_fisher_vector(gmm_list, tdd, output_file)
    return True


# python extract_fv.py -d /home/zhenyang/Workspace/data/UCF101/list_UCF101.txt -m rgb -l conv5 -c 3 -n 2 -s 1 -t 13320
# python extract_fv.py -d /home/zhenyang/Workspace/data/UCF101/list_UCF101.txt -m flow -l conv5 -c 3 -n 2 -s 1 -t 13320
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FisherExtractior')
    parser.add_argument('-d', '--dataset', dest='dataset', help='Specify dataset to process.', type=str, required=False)
    parser.add_argument('-s', '--startvid', dest='startvid', help='Specify video id start to process.', type=int, required=False)
    parser.add_argument('-t', '--tovid', dest='tovid', help='Specify video id until to process.', type=int, required=False)
    parser.add_argument('-m', '--mode', dest='mode', help='Specify mode, spatial network (rgb) or temporal network (flow).', type=str, required=True)
    parser.add_argument('-l', '--layer', dest='layer', help='Specify layer to extract features.', type=str, required=True)
    parser.add_argument('-c', '--scale', dest='scale', help='Specify scale to extract features.', type=int, required=True)
    parser.add_argument('-n', '--norm', dest='norm', help='Specify norm to extract features.', type=int, required=True)
    # parser.add_argument('-p', '--power', dest='power', help='Specify power to extract features.', type=float)

    args = parser.parse_args()

    if args.dataset is None:
        print 'Not specify dataset, using UCF101 by default...'
        args.dataset = '/home/zhenyang/Workspace/data/UCF101/list_UCF101.txt'

    print '***************************************'
    print '*********** EXTRACT FISHER ************'
    print '***************************************'
    print 'Dataset: %s' % (args.dataset, )

    base_dir = os.path.dirname(args.dataset)
    # tdd_flow_conv5_scale_3_norm_2
    feature = 'tdd_%s_%s_scale_%d_norm_%d' % (args.mode, args.layer, args.scale, args.norm)
    print 'Feature: %s' % (feature, )

    TDD_DIR = os.path.join(base_dir, 'features', feature)
    FV_DIR = os.path.join(base_dir, 'features', 'fv_'+feature)
    # UCF101_gmm256_pca64_tdd_flow_conv5_scale_3_norm_2_power_0.5.npz
    gmm_file = os.path.join(base_dir, 'features', 'gmm', 'UCF101_gmm256_pca64_%s_power_0.5' % (feature,))
    print 'GMM: %s' % (gmm_file, )

    f = open(args.dataset, 'r')
    input_videos = f.readlines()
    f.close()
    input_videos = [line.split()[0] for line in [video.rstrip() for video in input_videos]]

    Nf = len(input_videos)
    startvid = 0
    toid = Nf
    if args.startvid is not None and args.tovid is not None:
        startvid = max([args.startvid-1, startvid])
        toid = min([args.tovid, toid])
    input_videos = input_videos[startvid:toid]

    ###Just to prevent overwriting already processed vids
    completed_vids = [filename.split('.')[0] for filename in os.listdir(FV_DIR) if filename.endswith('.fv.mat')]
    overlap = [vid for vid in input_videos if vid.split('.')[0] in completed_vids]

    gmm_list = np.load(gmm_file+".npz")['gmm_list']
    #Multi-threaded FV construction.
    numThreads = 10
    pool = ThreadPool.ThreadPool(numThreads)
    for vid in input_videos:
        if vid not in overlap:
            #processVideo(vid,TDD_DIR,FV_DIR,gmm_list)
            pool.add_task(processVideo,vid,TDD_DIR,FV_DIR,gmm_list)
    pool.wait_completion()

    print '*********** PROCESSED ALL *************'

