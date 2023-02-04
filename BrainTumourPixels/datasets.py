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
# import nibabel as nib
# import imageio
#
# brain_data_path = "/root/regression_uncertainty/datasets/"
#
#
#
#
# filenames = os.listdir(brain_data_path + "Task01_BrainTumour/imagesTr")
# filenames = [filename for filename in filenames if filename[0] != "."]
#
# filenames_test = [filename for filename in filenames if int(filename.split("BRATS_")[1].split(".nii")[0]) >= 390]
# filenames_val = [filename for filename in filenames if (int(filename.split("BRATS_")[1].split(".nii")[0]) >= 295) and (int(filename.split("BRATS_")[1].split(".nii")[0]) < 390)]
# filenames_train = [filename for filename in filenames if (filename not in filenames_test) and (filename not in filenames_val)]
#
# print ("num 3D volumes for train: %d" % len(filenames_train))
# print ("num 3D volumes for val: %d" % len(filenames_val))
# print ("num 3D volumes test: %d" % len(filenames_test))
#
# if not os.path.exists(brain_data_path + "BrainTumourPixels"):
#     os.makedirs(brain_data_path + "BrainTumourPixels")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/train"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/train")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/train/slices"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/train/slices")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/train/mask_slices"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/train/mask_slices")
# for (file_i, filename) in enumerate(filenames_train):
#     volume = nib.load(brain_data_path + "Task01_BrainTumour/imagesTr/" + filename).get_fdata() # (shape: (240, 240, 155, 4))
#     mask_volume = nib.load(brain_data_path + "Task01_BrainTumour/labelsTr/" + filename).get_fdata() # (shape: (240, 240, 155)) (0: background, 1-3: tumour)
#
#     for i in range(volume.shape[2]):
#         slice = volume[:, :, i, 0:3] # (shape: (240, 240, 3))
#         mask_slice = mask_volume[:, :, i] # (shape: (240, 240)) (shape: (512, 512)) (0: background, 1-3: tumour)
#         num_brain_pixels = np.sum(mask_slice > 0)
#         if num_brain_pixels > 0:
#             imageio.imwrite(brain_data_path + "BrainTumourPixels/train/slices/" + "%s_slice_%d.png" % (filename.split(".nii.gz")[0], i), slice)
#             imageio.imwrite(brain_data_path + "BrainTumourPixels/train/mask_slices/" + "%s_slice_%d.png" % (filename.split(".nii.gz")[0], i), mask_slice)
#
# if not os.path.exists(brain_data_path + "BrainTumourPixels"):
#     os.makedirs(brain_data_path + "BrainTumourPixels")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/val"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/val")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/val/slices"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/val/slices")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/val/mask_slices"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/val/mask_slices")
# for (file_i, filename) in enumerate(filenames_val):
#     volume = nib.load(brain_data_path + "Task01_BrainTumour/imagesTr/" + filename).get_fdata() # (shape: (240, 240, 155, 4))
#     mask_volume = nib.load(brain_data_path + "Task01_BrainTumour/labelsTr/" + filename).get_fdata() # (shape: (240, 240, 155)) (0: background, 1-3: tumour)
#
#     for i in range(volume.shape[2]):
#         slice = volume[:, :, i, 0:3] # (shape: (240, 240, 3))
#         mask_slice = mask_volume[:, :, i] # (shape: (240, 240)) (shape: (512, 512)) (0: background, 1-3: tumour)
#         num_brain_pixels = np.sum(mask_slice > 0)
#         if num_brain_pixels > 0:
#             imageio.imwrite(brain_data_path + "BrainTumourPixels/val/slices/" + "%s_slice_%d.png" % (filename.split(".nii.gz")[0], i), slice)
#             imageio.imwrite(brain_data_path + "BrainTumourPixels/val/mask_slices/" + "%s_slice_%d.png" % (filename.split(".nii.gz")[0], i), mask_slice)
#
# if not os.path.exists(brain_data_path + "BrainTumourPixels"):
#     os.makedirs(brain_data_path + "BrainTumourPixels")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/test"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/test")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/test/slices"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/test/slices")
# if not os.path.exists(brain_data_path + "BrainTumourPixels/test/mask_slices"):
#     os.makedirs(brain_data_path + "BrainTumourPixels/test/mask_slices")
# for (file_i, filename) in enumerate(filenames_test):
#     volume = nib.load(brain_data_path + "Task01_BrainTumour/imagesTr/" + filename).get_fdata() # (shape: (240, 240, 155, 4))
#     mask_volume = nib.load(brain_data_path + "Task01_BrainTumour/labelsTr/" + filename).get_fdata() # (shape: (240, 240, 155)) (0: background, 1-3: tumour)
#
#     for i in range(volume.shape[2]):
#         slice = volume[:, :, i, 0:3] # (shape: (240, 240, 3))
#         mask_slice = mask_volume[:, :, i] # (shape: (240, 240)) (shape: (512, 512)) (0: background, 1-3: tumour)
#         num_brain_pixels = np.sum(mask_slice > 0)
#         if num_brain_pixels > 0:
#             imageio.imwrite(brain_data_path + "BrainTumourPixels/test/slices/" + "%s_slice_%d.png" % (filename.split(".nii.gz")[0], i), slice)
#             imageio.imwrite(brain_data_path + "BrainTumourPixels/test/mask_slices/" + "%s_slice_%d.png" % (filename.split(".nii.gz")[0], i), mask_slice)
#
#
#
#
# filenames_train = os.listdir(brain_data_path + "BrainTumourPixels/train/slices")
# filenames_val = os.listdir(brain_data_path + "BrainTumourPixels/val/slices")
# filenames_test = os.listdir(brain_data_path + "BrainTumourPixels/test/slices")
#
# imgs_train = []
# labels_train = []
# for i, filename in enumerate(filenames_train):
#     if i % 100 == 0:
#         print (i)
#
#     img = cv2.imread(brain_data_path + "BrainTumourPixels/train/slices/" + filename) # (shape: (240, 240, 3))
#     img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
#
#     mask = cv2.imread(brain_data_path + "BrainTumourPixels/train/mask_slices/" + filename) # (shape: (240, 240, 3))
#     mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST) # (shape: (64, 64, 3))
#
#     num_brain_pixels = np.sum(mask[:, :, 0] > 0)
#     if num_brain_pixels > 0:
#         imgs_train.append(img)
#         labels_train.append(num_brain_pixels)
#
# num_imgs_train = len(imgs_train)
# print ("num imgs train: %d" % num_imgs_train)
#
# images_train = np.zeros((num_imgs_train, 64, 64, 3), dtype=np.uint8)
# for i in range(num_imgs_train):
#     img = imgs_train[i]
#     images_train[i] = img
# labels_train = np.array(labels_train).astype(np.float32)
#
#
# imgs_val = []
# labels_val = []
# for i, filename in enumerate(filenames_val):
#     if i % 100 == 0:
#         print (i)
#
#     img = cv2.imread(brain_data_path + "BrainTumourPixels/val/slices/" + filename) # (shape: (240, 240, 3))
#     img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
#
#     mask = cv2.imread(brain_data_path + "BrainTumourPixels/val/mask_slices/" + filename) # (shape: (240, 240, 3))
#     mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST) # (shape: (64, 64, 3))
#
#     num_brain_pixels = np.sum(mask[:, :, 0] > 0)
#     if num_brain_pixels > 0:
#         imgs_val.append(img)
#         labels_val.append(num_brain_pixels)
#
# num_imgs_val = len(imgs_val)
# print ("num imgs val: %d" % num_imgs_val)
#
# images_val = np.zeros((num_imgs_val, 64, 64, 3), dtype=np.uint8)
# for i in range(num_imgs_val):
#     img = imgs_val[i]
#     images_val[i] = img
# labels_val = np.array(labels_val).astype(np.float32)
#
#
# imgs_test = []
# labels_test = []
# for i, filename in enumerate(filenames_test):
#     if i % 100 == 0:
#         print (i)
#
#     img = cv2.imread(brain_data_path + "BrainTumourPixels/test/slices/" + filename) # (shape: (240, 240, 3))
#     img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
#
#     mask = cv2.imread(brain_data_path + "BrainTumourPixels/test/mask_slices/" + filename) # (shape: (240, 240, 3))
#     mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST) # (shape: (64, 64, 3))
#
#     num_brain_pixels = np.sum(mask[:, :, 0] > 0)
#     if num_brain_pixels > 0:
#         imgs_test.append(img)
#         labels_test.append(num_brain_pixels)
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
# with open("/root/regression_uncertainty/BrainTumourPixels/labels_train.pkl", "wb") as file:
#     pickle.dump(labels_train, file)
# with open("/root/regression_uncertainty/BrainTumourPixels/images_train.pkl", "wb") as file:
#     pickle.dump(images_train, file)
#
# with open("/root/regression_uncertainty/BrainTumourPixels/labels_val.pkl", "wb") as file:
#     pickle.dump(labels_val, file)
# with open("/root/regression_uncertainty/BrainTumourPixels/images_val.pkl", "wb") as file:
#     pickle.dump(images_val, file)
#
# with open("/root/regression_uncertainty/BrainTumourPixels/labels_test.pkl", "wb") as file:
#     pickle.dump(labels_test, file)
# with open("/root/regression_uncertainty/BrainTumourPixels/images_test.pkl", "wb") as file:
#     pickle.dump(images_test, file)
# ################################################################################

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/BrainTumourPixels/labels_train.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/BrainTumourPixels/images_train.pkl", "rb") as file: # (needed for python3)
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
        with open("/root/regression_uncertainty/BrainTumourPixels/labels_val.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/BrainTumourPixels/images_val.pkl", "rb") as file: # (needed for python3)
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
        with open("/root/regression_uncertainty/BrainTumourPixels/labels_test.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/BrainTumourPixels/images_test.pkl", "rb") as file: # (needed for python3)
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
