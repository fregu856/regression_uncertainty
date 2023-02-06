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
import csv

ham10000_data_path = "/root/regression_uncertainty/datasets/ham10000/"


num_male = 0
num_female = 0
loc_to_num = {}
dataset_to_num = {}
with open(ham10000_data_path + "HAM10000_metadata", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader) # (ship the header row)
    for (step, row) in enumerate(reader):
        sex = row[5]
        if sex == "male":
            num_male += 1
        else:
            num_female += 1

        loc = row[6]
        if loc not in loc_to_num:
            loc_to_num[loc] = 0
        else:
            loc_to_num[loc] += 1

        dataset = row[7]
        if dataset not in dataset_to_num:
            dataset_to_num[dataset] = 0
        else:
            dataset_to_num[dataset] += 1
print (num_male)
print (num_female)
print (loc_to_num)
print (dataset_to_num)
print ("######################################################################")


img_ids_trainval = []
img_ids_test = []
with open(ham10000_data_path + "HAM10000_metadata", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader) # (ship the header row)
    for (step, row) in enumerate(reader):
        img_id = row[1]

        dataset = row[7]
        if dataset == "rosendahl":
            img_ids_test.append(img_id)
        else:
            img_ids_trainval.append(img_id)
num_imgs_trainval = len(img_ids_trainval)
num_imgs_test = len(img_ids_test)
print ("num imgs trainval: %d" % num_imgs_trainval)
print ("num imgs test: %d" % num_imgs_test)

np.random.shuffle(img_ids_trainval)
np.random.shuffle(img_ids_trainval)
np.random.shuffle(img_ids_trainval)
np.random.shuffle(img_ids_trainval)
img_ids_train = img_ids_trainval[0:int(0.85*num_imgs_trainval)]
img_ids_val = img_ids_trainval[int(0.85*num_imgs_trainval):]
num_imgs_train = len(img_ids_train)
num_imgs_val = len(img_ids_val)
print ("num imgs train: %d" % num_imgs_train)
print ("num imgs val: %d" % num_imgs_val)


images_train = np.zeros((num_imgs_train, 64, 64, 3), dtype=np.uint8)
labels_train = []
for (i, img_id) in enumerate(img_ids_train):
    if i % 100 == 0:
        print (i)

    img = cv2.imread(ham10000_data_path + img_id + ".jpg") # (shape: (450, 600, 3))
    # cv2.imwrite("/root/regression_uncertainty/SkinLesionPixels/before_img.png", img)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
    # cv2.imwrite("/root/regression_uncertainty/SkinLesionPixels/after_img.png", img)
    images_train[i] = img

    mask = cv2.imread(ham10000_data_path + "HAM10000_segmentations_lesion_tschandl/" + img_id + "_segmentation.png") # (shape: (450, 600, 3))
    # cv2.imwrite("/root/regression_uncertainty/SkinLesionPixels/before_mask.png", mask)
    mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST) # (shape: (64, 64, 3))
    # cv2.imwrite("/root/regression_uncertainty/SkinLesionPixels/after_mask.png", mask)
    num_lesion_pixels = np.sum(mask[:, :, 0] > 0)
    labels_train.append(num_lesion_pixels)
labels_train = np.array(labels_train).astype(np.float32)

images_val = np.zeros((num_imgs_val, 64, 64, 3), dtype=np.uint8)
labels_val = []
for (i, img_id) in enumerate(img_ids_val):
    if i % 100 == 0:
        print (i)

    img = cv2.imread(ham10000_data_path + img_id + ".jpg") # (shape: (450, 600, 3))
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
    images_val[i] = img

    mask = cv2.imread(ham10000_data_path + "HAM10000_segmentations_lesion_tschandl/" + img_id + "_segmentation.png") # (shape: (450, 600, 3))
    mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST) # (shape: (64, 64, 3))
    num_lesion_pixels = np.sum(mask[:, :, 0] > 0)
    labels_val.append(num_lesion_pixels)
labels_val = np.array(labels_val).astype(np.float32)

images_test = np.zeros((num_imgs_test, 64, 64, 3), dtype=np.uint8)
labels_test = []
for (i, img_id) in enumerate(img_ids_test):
    if i % 100 == 0:
        print (i)

    img = cv2.imread(ham10000_data_path + img_id + ".jpg") # (shape: (450, 600, 3))
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
    images_test[i] = img

    mask = cv2.imread(ham10000_data_path + "HAM10000_segmentations_lesion_tschandl/" + img_id + "_segmentation.png") # (shape: (450, 600, 3))
    mask = cv2.resize(mask, (64, 64), interpolation=cv2.INTER_NEAREST) # (shape: (64, 64, 3))
    num_lesion_pixels = np.sum(mask[:, :, 0] > 0)
    labels_test.append(num_lesion_pixels)
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

with open("/root/regression_uncertainty/SkinLesionPixels/labels_train.pkl", "wb") as file:
    pickle.dump(labels_train, file)
with open("/root/regression_uncertainty/SkinLesionPixels/images_train.pkl", "wb") as file:
    pickle.dump(images_train, file)

with open("/root/regression_uncertainty/SkinLesionPixels/labels_val.pkl", "wb") as file:
    pickle.dump(labels_val, file)
with open("/root/regression_uncertainty/SkinLesionPixels/images_val.pkl", "wb") as file:
    pickle.dump(images_val, file)

with open("/root/regression_uncertainty/SkinLesionPixels/labels_test.pkl", "wb") as file:
    pickle.dump(labels_test, file)
with open("/root/regression_uncertainty/SkinLesionPixels/images_test.pkl", "wb") as file:
    pickle.dump(images_test, file)
################################################################################
