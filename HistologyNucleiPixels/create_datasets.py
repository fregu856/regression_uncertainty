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
import scipy.io as sio
import os

hist_nuclei_data_path = "/root/regression_uncertainty/datasets/SkinLesionPixels/"


trainval_img_paths = []
trainval_labelmat_paths = []
for i in range(1, 28): # (1, 2, ..., 27)
    img_path = hist_nuclei_data_path + "CoNSeP/Train/Images/train_%d.png" % i
    trainval_img_paths.append(img_path)

    labelmat_path = hist_nuclei_data_path + "CoNSeP/Train/Labels/train_%d.mat" % i
    trainval_labelmat_paths.append(labelmat_path)

for i in range(1, 15): # (1, 2, ..., 14)
    img_path = hist_nuclei_data_path + "CoNSeP/Test/Images/test_%d.png" % i
    trainval_img_paths.append(img_path)

    labelmat_path = hist_nuclei_data_path + "CoNSeP/Test/Labels/test_%d.mat" % i
    trainval_labelmat_paths.append(labelmat_path)

kumar_img_filenames_train = os.listdir(hist_nuclei_data_path + "kumar/train/Images") # ['TCGA-49-4488-01Z-00-DX1.tif', ..., 'TCGA-B0-5711-01Z-00-DX1.tif']
for img_filename in kumar_img_filenames_train:
    img_path = hist_nuclei_data_path + "kumar/train/Images/" + img_filename
    trainval_img_paths.append(img_path)

    labelmat_path = hist_nuclei_data_path + "kumar/train/Labels/" + img_filename.split(".tif")[0] + ".mat"
    trainval_labelmat_paths.append(labelmat_path)

kumar_img_filenames_test_same = os.listdir(hist_nuclei_data_path + "kumar/test_same/Images")
for img_filename in kumar_img_filenames_test_same:
    img_path = hist_nuclei_data_path + "kumar/test_same/Images/" + img_filename
    trainval_img_paths.append(img_path)

    labelmat_path = hist_nuclei_data_path + "kumar/test_same/Labels/" + img_filename.split(".tif")[0] + ".mat"
    trainval_labelmat_paths.append(labelmat_path)

kumar_img_filenames_test_diff = os.listdir(hist_nuclei_data_path + "kumar/test_diff/Images")
for img_filename in kumar_img_filenames_test_diff:
    img_path = hist_nuclei_data_path + "kumar/test_diff/Images/" + img_filename
    trainval_img_paths.append(img_path)

    labelmat_path = hist_nuclei_data_path + "kumar/test_diff/Labels/" + img_filename.split(".tif")[0] + ".mat"
    trainval_labelmat_paths.append(labelmat_path)

print ("num original size hist imgs for trainval: %d" % len(trainval_img_paths))

imgs_trainval = []
labels_trainval = []
for id in range(len(trainval_img_paths)):
    print (id)

    img = cv2.imread(trainval_img_paths[id]) # (shape: (1000, 1000, 3))
    # cv2.imwrite("/root/regression_uncertainty/HistologyNucleiPixels/trainval_origimg_%d.png" % id, img)

    inst_map = sio.loadmat(trainval_labelmat_paths[id])["inst_map"] # (shape: (1000, 1000)) (0: background pixel, 1: pixel of first nuclei instance, 2: pixel of second nuclei instance, ...)
    mask = np.where(inst_map > 0, 1, 0)
    # mask_img = np.where(mask == 1, 255, 0)
    # cv2.imwrite("/root/regression_uncertainty/HistologyNucleiPixels/trainval_origimg_mask_%d.png" % id, mask_img)

    for i in range(int(img.shape[0]/64)):
        for j in range(int(img.shape[1]/64)):
            patch_img = img[(i*64):((i+1)*64), (j*64):((j+1)*64), :] # (shape: (64, 64, 3))
            # cv2.imwrite("/root/regression_uncertainty/HistologyNucleiPixels/trainval_origimg_%d_patch_%d_%d.png" % (id, i, j), patch_img)

            patch_mask = mask[(i*64):((i+1)*64), (j*64):((j+1)*64)] # (shape: (64, 64))
            # patch_mask_img = np.where(patch_mask == 1, 255, 0)
            # cv2.imwrite("/root/regression_uncertainty/HistologyNucleiPixels/trainval_origimg_mask_%d_patch_%d_%d.png" % (id, i, j), patch_mask_img)

            num_nuclei_pixels = np.sum(patch_mask > 0)
            if num_nuclei_pixels > 0:
                imgs_trainval.append(patch_img)
                labels_trainval.append(num_nuclei_pixels)

num_trainval_imgs = len(imgs_trainval)
print ("num imgs trainval: %d" % num_trainval_imgs)
print (np.min(np.array(labels_trainval)))
print (np.max(np.array(labels_trainval)))
print (np.mean(np.array(labels_trainval)))

trainval_ids = list(range(num_trainval_imgs))
np.random.shuffle(trainval_ids)
np.random.shuffle(trainval_ids)
np.random.shuffle(trainval_ids)
np.random.shuffle(trainval_ids)
train_ids = trainval_ids[0:int(0.8*num_trainval_imgs)]
val_ids = trainval_ids[int(0.8*num_trainval_imgs):]

num_imgs_train = len(train_ids)
num_imgs_val = len(val_ids)
print ("num imgs train: %d" % num_imgs_train)
print ("num imgs val: %d" % num_imgs_val)

images_train = np.zeros((num_imgs_train, 64, 64, 3), dtype=np.uint8)
labels_train = []
for i in range(num_imgs_train):
    id = train_ids[i]

    img = imgs_trainval[id]
    images_train[i] = img

    label = labels_trainval[id]
    labels_train.append(label)
labels_train = np.array(labels_train).astype(np.float32)

images_val = np.zeros((num_imgs_val, 64, 64, 3), dtype=np.uint8)
labels_val = []
for i in range(num_imgs_val):
    id = val_ids[i]

    img = imgs_trainval[id]
    images_val[i] = img

    label = labels_trainval[id]
    labels_val.append(label)
labels_val = np.array(labels_val).astype(np.float32)




test_img_paths = []
test_labelmat_paths = []
tnbc_img_filenames = os.listdir(hist_nuclei_data_path + "tnbc/Images/5784")
for img_filename in tnbc_img_filenames:
    img_path = hist_nuclei_data_path + "tnbc/Images/5784/" + img_filename
    test_img_paths.append(img_path)

    labelmat_path = hist_nuclei_data_path + "tnbc/Labels/" + img_filename.split(".png")[0] + ".mat"
    test_labelmat_paths.append(labelmat_path)

print ("num original size hist imgs for test: %d" % len(test_img_paths))

imgs_test = []
labels_test = []
for id in range(len(test_img_paths)):
    print (id)

    img = cv2.imread(test_img_paths[id]) # (shape: (512, 512, 3))
    # cv2.imwrite("/root/regression_uncertainty/HistologyNucleiPixels/test_origimg_%d.png" % id, img)

    inst_map = sio.loadmat(test_labelmat_paths[id])["inst_map"] # (shape: (512, 512)) (0: background pixel, 1: pixel of first nuclei instance, 2: pixel of second nuclei instance, ...)
    mask = np.where(inst_map > 0, 1, 0)
    # mask_img = np.where(mask == 1, 255, 0)
    # cv2.imwrite("/root/regression_uncertainty/HistologyNucleiPixels/test_origimg_mask_%d.png" % id, mask_img)

    for i in range(int(img.shape[0]/64)):
        for j in range(int(img.shape[1]/64)):
            patch_img = img[(i*64):((i+1)*64), (j*64):((j+1)*64), :] # (shape: (64, 64, 3))
            # cv2.imwrite("/root/regression_uncertainty/HistologyNucleiPixels/test_origimg_%d_patch_%d_%d.png" % (id, i, j), patch_img)

            patch_mask = mask[(i*64):((i+1)*64), (j*64):((j+1)*64)] # (shape: (64, 64))
            # patch_mask_img = np.where(patch_mask == 1, 255, 0)
            # cv2.imwrite("/root/regression_uncertainty/HistologyNucleiPixels/test_origimg_mask_%d_patch_%d_%d.png" % (id, i, j), patch_mask_img)

            num_nuclei_pixels = np.sum(patch_mask > 0)
            if num_nuclei_pixels > 0:
                imgs_test.append(patch_img)
                labels_test.append(num_nuclei_pixels)

num_imgs_test = len(imgs_test)
print ("num imgs test: %d" % num_imgs_test)

images_test = np.zeros((num_imgs_test, 64, 64, 3), dtype=np.uint8)
for i in range(num_imgs_test):
    img = imgs_test[i]
    images_test[i] = img
labels_test = np.array(labels_test).astype(np.float32)




print (labels_train.shape)
print (images_train.shape)
print (labels_val.shape)
print (images_val.shape)
print (labels_test.shape)
print (images_test.shape)

print (np.min(labels_train))
print (np.max(labels_train))
print (np.mean(labels_train))

print (np.min(labels_val))
print (np.max(labels_val))
print (np.mean(labels_val))

print (np.min(labels_test))
print (np.max(labels_test))
print (np.mean(labels_test))

with open("/root/regression_uncertainty/HistologyNucleiPixels/labels_train.pkl", "wb") as file:
    pickle.dump(labels_train, file)
with open("/root/regression_uncertainty/HistologyNucleiPixels/images_train.pkl", "wb") as file:
    pickle.dump(images_train, file)

with open("/root/regression_uncertainty/HistologyNucleiPixels/labels_val.pkl", "wb") as file:
    pickle.dump(labels_val, file)
with open("/root/regression_uncertainty/HistologyNucleiPixels/images_val.pkl", "wb") as file:
    pickle.dump(images_val, file)

with open("/root/regression_uncertainty/HistologyNucleiPixels/labels_test.pkl", "wb") as file:
    pickle.dump(labels_test, file)
with open("/root/regression_uncertainty/HistologyNucleiPixels/images_test.pkl", "wb") as file:
    pickle.dump(images_test, file)
################################################################################
