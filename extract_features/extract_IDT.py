#!/usr/bin/python

import argparse
import os
import errno
import subprocess
import shutil
from os import listdir
from os.path import isfile, join
import json


def getVideoIDTFeatures(inputfile,outputfile):
    print '(1/1) getVideoIDTFeatures: ' + inputfile
    #'./DenseTrackStab -f vid_name -o vid_name(1:end-4).bin'
    command = './DenseTrackStab -f %s -o %s' % (inputfile,outputfile, )
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
    print '******** EXTRACT FEATURES **********'
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

    for i in range(startvid, toid):

        filename = filenames[i]
        filename_ = os.path.splitext(filename)[0]
        print 'Processing (%d/%d): %s' % (i+1,Nf,filename, )

        inputfile = os.path.join(base_dir, 'videos', filename)
        outputfile = os.path.join(base_dir, 'features', 'idt', filename_+'.bin')

        if not os.path.exists(outputfile):
            getVideoIDTFeatures(inputfile,outputfile)

    print '********* PROCESSED ALL ************'

