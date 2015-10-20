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
    command = './DenseTrackStab -f %s -o %s' % (inputfile, outputfile, )
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
    while proc.poll() is None:
        line = proc.stdout.readline()
        print(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FeatureExtractior')    
    parser.add_argument('-d', '--dataset', dest='dataset', help='Specify dataset to process.', type=str, required=False)
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
    with open(args.files) as fp:
        for line in fp:
            filenames.append(line.strip())

    Nf = len(filenames)
    for i in range(0, Nf):

        filename = filenames[i]
        print 'Processing (%d/%d): %s' % (i,Nf,filename, )

        inputfile = os.path.join(base_dir, 'videos', filename)
        outputfile = os.path.join(base_dir, 'features', 'idt', filename+'.bin')

        if not os.path.exists(outputfile):
            getVideoIDTFeatures(inputfile,outputfile)

    print '********* PROCESSED ALL ************'

