import numpy as np
import sys, os, random
import scipy.io
from yael import ynumpy
from tempfile import TemporaryFile
import argparse

"""
Can execute this as a script to populate the GMM or load it as a module

PCA reduction on each descriptor is set to false by default.
"""

# Path to the video repository
UCF101_DIR = "/home/zhenyang/Workspace/data/UCF101"


def populate_gmms(TDD_DIR, sample_vids, gmm_file, k_gmm, sample_size=1500000, PCA=0.5):
    """
    sample_size is the number of TDDs that we sample from the sample_vids.

    gmm_file is the output file to save the list of GMMs.
    Saves the GMMs in the gmm_file file as the gmm_list attribute.

    Returns the list of gmms.
    """
    nr_vids = len(sample_vids)
    nr_samples_pvid = int(np.ceil(sample_size/nr_vids))

    tdds = []
    for vid in sample_vids:
        if os.path.exists(os.path.join(TDD_DIR,vid)):
            data = scipy.io.loadmat(os.path.join(TDD_DIR,vid))
            nr_points = data['idt_cnn_feature'].shape[1]
            sample_size = min(nr_points,nr_samples_pvid)
            idx_sampled = random.sample(xrange(nr_points),sample_size)
            idx_sampled.sort()
            points_sampled = data['idt_cnn_feature'][:,idx_sampled]
            tdds.append(points_sampled.T)

    tdds = np.vstack(tdds)
    # save all sampled descriptors for learning gmm
    bm_file = os.path.join(os.path.dirname(gmm_file), 'bm_TDD_descriptors_%d_%s' % (sample_size,gmm_file))
    np.savez(bm_file, tdds=tdds)

    # why sqrt? just like root sift! power: 0.5
    tdds = np.sqrt(tdds)

    # Construct gmm models for each of the different descriptor types.
    gmm_list = gmm_model(tdds, k_gmm, PCA=PCA)
    np.savez(gmm_file, gmm_list=gmm_list)

    return gmm_list


def gmm_model(sample, k_gmm, PCA=0.5):
    """
    Returns a tuple: (gmm,mean,pca_transform)
    gmm is the ynumpy gmm model fro the sample data. 
    pca_tranform is None if PCA is True.
    Reduces the dimensions of the sample (by 50%) if PCA is true
    """

    print "Building GMM model"
    # convert to float32
    sample = sample.astype('float32')
    # compute mean and covariance matrix for the PCA
    mean = sample.mean(axis = 0) # for rows
    sample = sample - mean
    pca_transform = None
    if PCA:
        cov = np.dot(sample.T, sample)

        # decide to keep 1/2 of the original components, so shape[1]/2
        # compute PCA matrix and keep only 1/2 of the dimensions.
        orig_comps = sample.shape[1]
        pca_dim = int(orig_comps*PCA)
        # eigvecs are normalized.
        eigvals, eigvecs = np.linalg.eig(cov)
        perm = eigvals.argsort() # sort by increasing eigenvalue 
        pca_transform = eigvecs[:, perm[orig_comps-pca_dim:orig_comps]]   # eigenvectors for the last half eigenvalues
        # transform sample with PCA (note that numpy imposes line-vectors,
        # so we right-multiply the vectors)
        sample = np.dot(sample, pca_transform)
    # train GMM
    gmm = ynumpy.gmm_learn(sample, k_gmm)
    toReturn = (gmm,mean,pca_transform)
    return toReturn


def sampleVids(vid_list, nr_pcls=1):
    """
    vid_list is a text file of video names and their corresponding
    class.
    This function reads the video names and creates a list with one video
    from each class.
    """
    f = open(vid_list, 'r')
    videos = f.readlines()
    f.close()
    videos = [video.rstrip() for video in videos]
    vid_dict = {}
    for line in videos:
        l = line.split()
        key = int(l[1])
        if key not in vid_dict:
            vid_dict[key] = []
        vid_dict[key].append(l[0])
    
    samples = []
    for k,v in vid_dict.iteritems():
        #samples.extend(v[:1])
        samples.extend(v[:min(nr_pcls,len(v))])
    return samples


#python gmm.py -k 256 -f tdd_rgb_conv5_scale_3_norm_2 -l train1.txt -o UCF101_gmm256_pca64_tdd_rgb_conv5_scale_3_norm_2_power_0.5 -p 0.125
#python gmm.py -k 256 -f tdd_flow_conv5_scale_3_norm_2 -l train1.txt -o UCF101_gmm256_pca64_tdd_flow_conv5_scale_3_norm_2_power_0.5 -p 0.125
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--gmmk', help="Number of GMM modes", type=int, required=True)
    parser.add_argument('-f', '--feature', help="Feature to encode fisher vector", type=str)
    parser.add_argument('-l', '--vidlist', help="List of input videos from which to sample", type=str)
    parser.add_argument('-o', '--gmmfile', help="Output file to save the list of gmms", type=str)
    parser.add_argument('-p', '--pca', help="Percent of original descriptor components to retain after PCA", type=float)
    args = parser.parse_args()

    print args.gmmk
    print args.feature
    print args.vidlist
    print args.gmmfile
    print args.pca

    TDD_DIR = os.path.join(UCF101_DIR, 'features', args.feature)
    vid_list = os.path.join(UCF101_DIR, args.vidlist)
    gmm_file = os.path.join(UCF101_DIR, 'gmm', args.gmmfile)

    # vid_samples = sampleVids(vid_list)
    # select all for GMM training
    f = open(vid_list, 'r')
    input_videos = f.readlines()
    f.close()
    vid_samples = [line.split()[0] for line in [video.rstrip() for video in input_videos]]

    videos = []
    for vidname in vid_samples:
        vidname_ = os.path.splitext(vidname)[0]
        videos.append(vidname_+'.mat')
    populate_gmms(TDD_DIR,videos,gmm_file,args.gmmk,args.pca)

