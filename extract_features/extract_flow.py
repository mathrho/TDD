#!/usr/bin/python

import argparse
import os
import errno
import subprocess
import shutil
from os import listdir
from os.path import isfile, join
import json


flowbin = './denseFlow_gpu'


def getVideoFlowFeatures(inputfile,outputfile):
    print '(1/1) getVideoFlowFeatures: ' + inputfile
    #'./denseFlow -f ',vid_name,' -x test/flow_x -y test/flow_y -b 20'
    #'./denseFlow_gpu -d 1 -f ',vid_name,' -x test/flow_x -y test/flow_y -b 20 -t 1 -d 3'
    command = '%s -f %s -x %s/flow_x -y %s/flow_y -i %s/image -b 20 -t 1 -d 2' % (flowbin,inputfile,outputfile,outputfile,outputfile, )
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    while proc.poll() is None:
        line = proc.stdout.readline()
        print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FeatureExtractior')    
    parser.add_argument('-d', '--dataset', dest='dataset', help='Specify dataset to process.', type=str, required=False)
    parser.add_argument('-s', '--startvid', dest='startvid', help='Specify video id start to process.', type=int, required=False)
    parser.add_argument('-t', '--tovid', dest='tovid', help='Specify video id until to process.', type=int, required=False)
    args = parser.parse_args()

    if args.dataset is None:
        print 'Not specify dataset, using UCF101 by default...'
        args.dataset = '/home/zhenyang/Workspace/data/UCF101/list_UCF101.txt'

    print '***************************************'
    print '********** EXTRACT FEATURES ***********'
    print '***************************************'
    print 'Dataset: %s' % (args.dataset, )

    base_dir = os.path.dirname(args.dataset)

    filenames = []
    with open(args.dataset) as fp:
        for line in fp:
            filenames.append(line.strip())

    Nf = len(filenames)
    startvid = 0
    tovid = Nf
    if args.startvid is not None and args.tovid is not None:
        startvid = max([args.startvid-1, startvid])
        tovid = min([args.tovid, tovid])

    for i in range(startvid, tovid):

        filename = filenames[i]
        filename_ = os.path.splitext(filename)[0]
        print 'Processing (%d/%d): %s' % (i+1,Nf,filename, )

        inputfile = os.path.join(base_dir, 'videos', filename)
        outputfile = os.path.join(base_dir, 'features', 'flow_tvl1', filename_)

        if not os.path.exists(outputfile):
            os.makedirs(outputfile)
            getVideoFlowFeatures(inputfile,outputfile)

    print '********* PROCESSED ALL ************'

