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

# ################################################################################
# # run this once to generate the train/val/test data:
# ################################################################################
# import os
#
# inria_aerial_data_path = "/root/regression_uncertainty/datasets/"
#
# filenames = os.listdir(inria_aerial_data_path + "AerialImageDataset/train/images")
#
# filenames_trainval = []
# filenames_test = []
# for filename in filenames:
#     if filename[0:7] == "tyrol-w":
#         filenames_test.append(filename)
#     else:
#         if (filename[0:6] == "austin") or (filename[0:7] == "chicago"):
#             filenames_trainval.append(filename)
#
# print ("num original size imgs for trainval: %d" % len(filenames_trainval))
# print ("num original size imgs for test: %d" % len(filenames_test))
#
# imgs_trainval = []
# labels_trainval = []
# for id in range(len(filenames_trainval)):
#     print (id)
#
#     img = cv2.imread(inria_aerial_data_path + "AerialImageDataset/train/images/" + filenames_trainval[id]) # (shape: (5000, 5000, 3))
#     img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_LINEAR) # (shape: (1000, 1000, 3))
#     # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/trainval_origimg_%d.png" % id, img)
#
#     mask = cv2.imread(inria_aerial_data_path + "AerialImageDataset/train/gt/" + filenames_trainval[id]) # (shape: (5000, 5000, 3))
#     mask = cv2.resize(mask, (1000, 1000), interpolation=cv2.INTER_NEAREST) # (shape: (1000, 1000, 3))
#     # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/trainval_origimg_mask_%d.png" % id, mask)
#     # mask_img_overlay = 0.5*img + 0.5*mask
#     # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/trainval_origimg_mask_overlay_%d.png" % id, mask_img_overlay.astype(np.uint8))
#
#     for i in range(int(img.shape[0]/64)):
#         for j in range(int(img.shape[1]/64)):
#             patch_img = img[(i*64):((i+1)*64), (j*64):((j+1)*64), :] # (shape: (64, 64, 3))
#             # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/trainval_origimg_%d_patch_%d_%d.png" % (id, i, j), patch_img)
#
#             patch_mask = mask[(i*64):((i+1)*64), (j*64):((j+1)*64)] # (shape: (64, 64, 3))
#             # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/trainval_origimg_mask_%d_patch_%d_%d.png" % (id, i, j), patch_mask)
#
#             num_building_pixels = np.sum(patch_mask[:, :, 0] > 0)
#             if num_building_pixels > 0:
#                 imgs_trainval.append(patch_img)
#                 labels_trainval.append(num_building_pixels)
#
# num_trainval_imgs = len(imgs_trainval)
# print ("num imgs trainval: %d" % num_trainval_imgs)
# print (np.min(np.array(labels_trainval)))
# print (np.max(np.array(labels_trainval)))
# print (np.mean(np.array(labels_trainval)))
#
# trainval_ids = list(range(num_trainval_imgs))
# np.random.shuffle(trainval_ids)
# np.random.shuffle(trainval_ids)
# np.random.shuffle(trainval_ids)
# np.random.shuffle(trainval_ids)
# train_ids = trainval_ids[0:int(0.8*num_trainval_imgs)]
# val_ids = trainval_ids[int(0.8*num_trainval_imgs):]
#
# num_imgs_train = len(train_ids)
# num_imgs_val = len(val_ids)
# print ("num imgs train: %d" % num_imgs_train)
# print ("num imgs val: %d" % num_imgs_val)
#
# images_train = np.zeros((num_imgs_train, 64, 64, 3), dtype=np.uint8)
# labels_train = []
# for i in range(num_imgs_train):
#     id = train_ids[i]
#
#     img = imgs_trainval[id]
#     images_train[i] = img
#
#     label = labels_trainval[id]
#     labels_train.append(label)
# labels_train = np.array(labels_train).astype(np.float32)
#
# images_val = np.zeros((num_imgs_val, 64, 64, 3), dtype=np.uint8)
# labels_val = []
# for i in range(num_imgs_val):
#     id = val_ids[i]
#
#     img = imgs_trainval[id]
#     images_val[i] = img
#
#     label = labels_trainval[id]
#     labels_val.append(label)
# labels_val = np.array(labels_val).astype(np.float32)
#
#
#
#
# imgs_test = []
# labels_test = []
# for id in range(len(filenames_test)):
#     print (id)
#
#     img = cv2.imread(inria_aerial_data_path + "AerialImageDataset/train/images/" + filenames_test[id]) # (shape: (5000, 5000, 3))
#     img = cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_LINEAR) # (shape: (1000, 1000, 3))
#     # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/test_origimg_%d.png" % id, img)
#
#     mask = cv2.imread(inria_aerial_data_path + "AerialImageDataset/train/gt/" + filenames_test[id]) # (shape: (5000, 5000, 3))
#     mask = cv2.resize(mask, (1000, 1000), interpolation=cv2.INTER_NEAREST) # (shape: (1000, 1000, 3))
#     # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/test_origimg_mask_%d.png" % id, mask)
#     # mask_img_overlay = 0.5*img + 0.5*mask
#     # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/test_origimg_mask_overlay_%d.png" % id, mask_img_overlay.astype(np.uint8))
#
#     for i in range(int(img.shape[0]/64)):
#         for j in range(int(img.shape[1]/64)):
#             patch_img = img[(i*64):((i+1)*64), (j*64):((j+1)*64), :] # (shape: (64, 64, 3))
#             # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/test_origimg_%d_patch_%d_%d.png" % (id, i, j), patch_img)
#
#             patch_mask = mask[(i*64):((i+1)*64), (j*64):((j+1)*64)] # (shape: (64, 64, 3))
#             # cv2.imwrite("/root/regression_uncertainty/AerialBuildingPixels/test_origimg_mask_%d_patch_%d_%d.png" % (id, i, j), patch_mask)
#
#             num_building_pixels = np.sum(patch_mask[:, :, 0] > 0)
#             if num_building_pixels > 0:
#                 imgs_test.append(patch_img)
#                 labels_test.append(num_building_pixels)
#
# num_imgs_test = len(imgs_test)
# print ("num imgs test: %d" % num_imgs_test)
#
# images_test = np.zeros((num_imgs_test, 64, 64, 3), dtype=np.uint8)
# for i in range(num_imgs_test):
#     img = imgs_test[i]
#     images_test[i] = img
# labels_test = np.array(labels_test).astype(np.float32)
#
#
#
#
# print (labels_train.shape)
# print (images_train.shape)
# print (labels_val.shape)
# print (images_val.shape)
# print (labels_test.shape)
# print (images_test.shape)
#
# print (np.min(labels_train))
# print (np.max(labels_train))
# print (np.mean(labels_train))
#
# print (np.min(labels_val))
# print (np.max(labels_val))
# print (np.mean(labels_val))
#
# print (np.min(labels_test))
# print (np.max(labels_test))
# print (np.mean(labels_test))
#
# with open("/root/regression_uncertainty/AerialBuildingPixels/labels_train.pkl", "wb") as file:
#     pickle.dump(labels_train, file)
# with open("/root/regression_uncertainty/AerialBuildingPixels/images_train.pkl", "wb") as file:
#     pickle.dump(images_train, file)
#
# with open("/root/regression_uncertainty/AerialBuildingPixels/labels_val.pkl", "wb") as file:
#     pickle.dump(labels_val, file)
# with open("/root/regression_uncertainty/AerialBuildingPixels/images_val.pkl", "wb") as file:
#     pickle.dump(images_val, file)
#
# with open("/root/regression_uncertainty/AerialBuildingPixels/labels_test.pkl", "wb") as file:
#     pickle.dump(labels_test, file)
# with open("/root/regression_uncertainty/AerialBuildingPixels/images_test.pkl", "wb") as file:
#     pickle.dump(images_test, file)
# ################################################################################

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/AerialBuildingPixels/labels_train.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/AerialBuildingPixels/images_train.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetTrain - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, label)

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/AerialBuildingPixels/labels_val.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/AerialBuildingPixels/images_val.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetVal - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, label)

    def __len__(self):
        return self.num_examples

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/AerialBuildingPixels/labels_test.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/AerialBuildingPixels/images_test.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, label)

    def __len__(self):
        return self.num_examples
