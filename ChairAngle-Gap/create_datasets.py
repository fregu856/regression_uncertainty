# camera-ready

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
import scipy.stats

import pickle

import cv2

################################################################################
# run this once to generate the train/val/test data:
################################################################################
import h5py
from tqdm import tqdm

hf = h5py.File("/root/regression_uncertainty/datasets/RC-49_64x64.h5", 'r')
labels = hf['labels'][:]
labels = labels.astype(np.float32)
images = hf['images'][:]
inds_train = hf['indx_train'][:]
hf.close()

print (images.shape)
print (labels.shape)
print (np.max(labels))
print (np.min(labels))
print ("############")

inds_test = []
for i in range(labels.shape[0]):
    if i not in inds_train:
        inds_test.append(i)

labels_train = labels[inds_train]
images_train = images[inds_train]
labels_test = labels[inds_test]
images_test = images[inds_test]
print (images_train.shape)
print (labels_train.shape)
print (np.max(labels_train))
print (np.min(labels_train))
print (images_test.shape)
print (labels_test.shape)
print (np.max(labels_test))
print (np.min(labels_test))
print ("############")

inds_filtered_train = []
for i in range(labels_train.shape[0]):
    if (labels_train[i] > 0) and (labels_train[i] < 90):
        if (labels_train[i] < 30) or (labels_train[i] > 60):
            inds_filtered_train.append(i)

inds_filtered_test = []
for i in range(labels_test.shape[0]):
    if (labels_test[i] > 0) and (labels_test[i] < 90):
        inds_filtered_test.append(i)

labels_train = labels_train[inds_filtered_train]
images_train = images_train[inds_filtered_train]
labels_test = labels_test[inds_filtered_test]
images_test = images_test[inds_filtered_test]
print (images_train.shape)
print (labels_train.shape)
print (np.max(labels_train))
print (np.min(labels_train))
print (images_test.shape)
print (labels_test.shape)
print (np.max(labels_test))
print (np.min(labels_test))
print ("############")

# for each angle, take no more than 25 images
image_num_threshold = 25
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images_test), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels_test))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels_test == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images_test = images_test[sel_indx]
labels_test = labels_test[sel_indx]
print (images_test.shape)
print (labels_test.shape)
print (np.max(labels_test))
print (np.min(labels_test))
print ("############")

labels_trainval = labels_train
images_trainval = images_train
inds_trainval = list(range(len(labels_trainval)))
np.random.shuffle(inds_trainval)
np.random.shuffle(inds_trainval)
np.random.shuffle(inds_trainval)
np.random.shuffle(inds_trainval)
inds_train = inds_trainval[0:int(0.8*len(inds_trainval))]
inds_val = inds_trainval[int(0.8*len(inds_trainval)):]
labels_train = labels_trainval[inds_train]
images_train = images_trainval[inds_train]
labels_val = labels_trainval[inds_val]
images_val = images_trainval[inds_val]

print (labels_train.shape)
print (images_train.shape)
print (labels_val.shape)
print (images_val.shape)
print (labels_test.shape)
print (images_test.shape)

with open("/root/regression_uncertainty/ChairAngle-Gap/labels_train.pkl", "wb") as file:
    pickle.dump(labels_train, file)
with open("/root/regression_uncertainty/ChairAngle-Gap/images_train.pkl", "wb") as file:
    pickle.dump(images_train, file)

with open("/root/regression_uncertainty/ChairAngle-Gap/labels_val.pkl", "wb") as file:
    pickle.dump(labels_val, file)
with open("/root/regression_uncertainty/ChairAngle-Gap/images_val.pkl", "wb") as file:
    pickle.dump(images_val, file)

with open("/root/regression_uncertainty/ChairAngle-Gap/labels_test.pkl", "wb") as file:
    pickle.dump(labels_test, file)
with open("/root/regression_uncertainty/ChairAngle-Gap/images_test.pkl", "wb") as file:
    pickle.dump(images_test, file)
################################################################################
