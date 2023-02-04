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
# img_filenames = os.listdir("/root/regression_uncertainty/datasets/Cell200")
# print (img_filenames)
# print (len(img_filenames))
#
# img = cv2.imread("/root/regression_uncertainty/datasets/Cell200/" + img_filenames[0], -1)
# print (img)
# print (img.shape)
# print (img.dtype)
#
# num_examples = len(img_filenames)
# print (num_examples)
#
# img_filenames_in = []
# img_filenames_out = []
# for img_filename in img_filenames:
#     label = float(img_filename.split("_")[1].split(".0.")[0])
#
#     if (label < 50) or (label >= 150):
#         img_filenames_in.append(img_filename)
#     else:
#         img_filenames_out.append(img_filename)
#
# print ("num examples in: %d" % len(img_filenames_in))
# print ("num examples out: %d" % len(img_filenames_out))
#
# np.random.shuffle(img_filenames_in)
# np.random.shuffle(img_filenames_in)
# np.random.shuffle(img_filenames_in)
# np.random.shuffle(img_filenames_in)
# #
# np.random.shuffle(img_filenames_out)
# np.random.shuffle(img_filenames_out)
# np.random.shuffle(img_filenames_out)
# np.random.shuffle(img_filenames_out)
#
# num_imgs_train = 10000
# num_imgs_val = 2000
# num_imgs_test = 10000
# img_filenames_train = img_filenames_in[0:num_imgs_train]
# img_filenames_val = img_filenames_in[num_imgs_train:(num_imgs_train+num_imgs_val)]
# img_filenames_test_1 = img_filenames_in[(num_imgs_train+num_imgs_val):((num_imgs_train+num_imgs_val)+int(num_imgs_test/2))]
# img_filenames_test_2 = img_filenames_out[0:int(num_imgs_test/2)]
# img_filenames_test = np.concatenate((img_filenames_test_1, img_filenames_test_2))
#
# print (len(img_filenames_train))
# print (len(img_filenames_val))
# print (len(img_filenames_test))
#
# images_train = np.zeros((num_imgs_train, img.shape[0], img.shape[1]), dtype=img.dtype)
# images_val = np.zeros((num_imgs_val, img.shape[0], img.shape[1]), dtype=img.dtype)
# images_test = np.zeros((num_imgs_test, img.shape[0], img.shape[1]), dtype=img.dtype)
# print (images_train.shape)
# print (images_train.dtype)
# print (images_val.shape)
# print (images_val.dtype)
# print (images_test.shape)
# print (images_test.dtype)
#
# labels_train = []
# for (i, img_filename) in enumerate(img_filenames_train):
#     img = cv2.imread("/root/regression_uncertainty/datasets/Cell200/" + img_filename, -1)
#     images_train[i] = img
#
#     label = float(img_filename.split("_")[1].split(".0.")[0])
#     labels_train.append(label)
# labels_train = np.array(labels_train).astype(np.float32)
#
# labels_val = []
# for (i, img_filename) in enumerate(img_filenames_val):
#     img = cv2.imread("/root/regression_uncertainty/datasets/Cell200/" + img_filename, -1)
#     images_val[i] = img
#
#     label = float(img_filename.split("_")[1].split(".0.")[0])
#     labels_val.append(label)
# labels_val = np.array(labels_val).astype(np.float32)
#
# labels_test = []
# for (i, img_filename) in enumerate(img_filenames_test):
#     img = cv2.imread("/root/regression_uncertainty/datasets/Cell200/" + img_filename, -1)
#     images_test[i] = img
#
#     label = float(img_filename.split("_")[1].split(".0.")[0])
#     labels_test.append(label)
# labels_test = np.array(labels_test).astype(np.float32)
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
# #
# print (np.min(labels_test))
# print (np.max(labels_test))
# print (np.mean(labels_test))
#
# with open("/root/regression_uncertainty/Cells-Gap/labels_train.pkl", "wb") as file:
#     pickle.dump(labels_train, file)
# with open("/root/regression_uncertainty/Cells-Gap/images_train.pkl", "wb") as file:
#     pickle.dump(images_train, file)
#
# with open("/root/regression_uncertainty/Cells-Gap/labels_val.pkl", "wb") as file:
#     pickle.dump(labels_val, file)
# with open("/root/regression_uncertainty/Cells-Gap/images_val.pkl", "wb") as file:
#     pickle.dump(images_val, file)
#
# with open("/root/regression_uncertainty/Cells-Gap/labels_test.pkl", "wb") as file:
#     pickle.dump(labels_test, file)
# with open("/root/regression_uncertainty/Cells-Gap/images_test.pkl", "wb") as file:
#     pickle.dump(images_test, file)
# ################################################################################

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/Cells-Gap/labels_train.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/Cells-Gap/images_train.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetTrain - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64))
        img = np.expand_dims(img, axis=2) # (shape: (64, 64, 1))
        img = img*np.ones((img.shape[0], img.shape[1], 3), dtype=img.dtype) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/Cells-Gap/labels_val.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/Cells-Gap/images_val.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetVal - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64))
        img = np.expand_dims(img, axis=2) # (shape: (64, 64, 1))
        img = img*np.ones((img.shape[0], img.shape[1], 3), dtype=img.dtype) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/Cells-Gap/labels_test.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/Cells-Gap/images_test.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        self.num_examples = self.labels.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)
        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (64, 64))
        img = np.expand_dims(img, axis=2) # (shape: (64, 64, 1))
        img = img*np.ones((img.shape[0], img.shape[1], 3), dtype=img.dtype) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 64, 64))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples
