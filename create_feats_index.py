from glob import glob
from feat_utils import *
from nn_utils import *
import numpy as np
import multiprocessing
import caffe

global net
global mean_im
global feat_layer

mean_im = load_mean_from_binaryproto('/project/focus/datasets/tc_tripletloss/mean.binaryproto')
net = load_net('/project/focus/abby/tc_tripletloss/models/deploy/deploy.prototxt','/project/focus/abby/tc_tripletloss/models/alexnet_places365.caffemodel')
feat_layer = 'fc8'

ims = glob('/project/focus/datasets/hotel_chains/images/3*.jpg')

allFeats = np.empty((len(ims),365))
for ix in range(len(ims)):
    print ix
    allFeats[ix,:] = extract_features(ims[ix],mean_im,net,feat_layer)

save_ims_and_index(ims,allFeats,'/project/focus/datasets/hotel_chains/features/')
