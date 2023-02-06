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

class DatasetTrain(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/ChairAngle-Tails/labels_train.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/ChairAngle-Tails/images_train.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

        self.num_examples = self.labels.shape[0]

        print ("DatasetTrain - number of images: %d" % self.num_examples)

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (3, 64, 64))
        img = np.transpose(img, (1, 2, 0)) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 200, 200))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples

class DatasetVal(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/ChairAngle-Tails/labels_val.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/ChairAngle-Tails/images_val.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

        self.num_examples = self.labels.shape[0]

        print ("DatasetVal - number of images: %d" % self.num_examples)

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (3, 64, 64))
        img = np.transpose(img, (1, 2, 0)) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 200, 200))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples

class DatasetTest(torch.utils.data.Dataset):
    def __init__(self):
        with open("/root/regression_uncertainty/ChairAngle-Tails/labels_test.pkl", "rb") as file: # (needed for python3)
            self.labels = pickle.load(file)
        with open("/root/regression_uncertainty/ChairAngle-Tails/images_test.pkl", "rb") as file: # (needed for python3)
            self.imgs = pickle.load(file)

        print (self.labels.shape)
        print (self.imgs.shape)

        print (np.min(self.labels))
        print (np.max(self.labels))
        print (np.mean(self.labels))

        self.num_examples = self.labels.shape[0]

        print ("DatasetTest - number of images: %d" % self.num_examples)

    def __getitem__(self, index):
        angle = self.labels[index]
        img = self.imgs[index] # (shape: (3, 64, 64))
        img = np.transpose(img, (1, 2, 0)) # (shape: (64, 64, 3))

        img = img/255.0
        img = img - np.array([0.485, 0.456, 0.406])
        img = img/np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1)) # (shape: (3, 200, 200))
        img = img.astype(np.float32)

        return (img, angle)

    def __len__(self):
        return self.num_examples
