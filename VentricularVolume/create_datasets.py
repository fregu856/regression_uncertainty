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

echonet_dynamic_data_path = "/root/regression_uncertainty/datasets"


video_ids_train = []
esvs_train = []
edvs_train = []
video_ids_val = []
esvs_val = []
edvs_val = []
video_ids_test = []
esvs_test = []
edvs_test = []
with open(echonet_dynamic_data_path + "/EchoNet-Dynamic/FileList.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader) # (ship the header row)
    for (step, row) in enumerate(reader):
        video_id = row[0]
        esv = float(row[2])
        edv = float(row[3])

        split = row[8]
        if split == "TRAIN":
            video_ids_train.append(video_id)
            esvs_train.append(esv)
            edvs_train.append(edv)
        elif split == "VAL":
            video_ids_val.append(video_id)
            esvs_val.append(esv)
            edvs_val.append(edv)
        elif split == "TEST":
            video_ids_test.append(video_id)
            esvs_test.append(esv)
            edvs_test.append(edv)

num_videos_train = len(video_ids_train)
num_videos_val = len(video_ids_val)
num_videos_test = len(video_ids_test)
print ("num videos train: %d" % num_videos_train)
print ("num videos val: %d" % num_videos_val)
print ("num videos test: %d" % num_videos_test)
print (np.min(np.array(esvs_train)))
print (np.max(np.array(esvs_train)))
print (np.mean(np.array(esvs_train)))
print (np.min(np.array(edvs_train)))
print (np.max(np.array(edvs_train)))
print (np.mean(np.array(edvs_train)))
print ("#####")
print (np.min(np.array(esvs_val)))
print (np.max(np.array(esvs_val)))
print (np.mean(np.array(esvs_val)))
print (np.min(np.array(edvs_val)))
print (np.max(np.array(edvs_val)))
print (np.mean(np.array(edvs_val)))
print ("#####")
print (np.min(np.array(esvs_test)))
print (np.max(np.array(esvs_test)))
print (np.mean(np.array(esvs_test)))
print (np.min(np.array(edvs_test)))
print (np.max(np.array(edvs_test)))
print (np.mean(np.array(edvs_test)))

video_id_to_frames = {}
with open(echonet_dynamic_data_path + "/EchoNet-Dynamic/VolumeTracings.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    next(reader) # (ship the header row)
    for (step, row) in enumerate(reader):
        video_id = row[0].split(".avi")[0]
        frame = int(row[5])

        if video_id not in video_id_to_frames:
            video_id_to_frames[video_id] = [frame]
        else:
            if frame not in video_id_to_frames[video_id]:
                video_id_to_frames[video_id].append(frame)

print (len(video_id_to_frames))




imgs_train = []
labels_train = []
for i in range(num_videos_train):
    if i % 100 == 0:
        print (i)

    video_id = video_ids_train[i]
    esv = esvs_train[i] # (smaller volume)
    edv = edvs_train[i] # (large volume)

    video_path = echonet_dynamic_data_path + "EchoNet-Dynamic/Videos/" + video_id + ".avi"

    if video_id in video_id_to_frames: # (a small number of videos apparently don't have tracings)
        frames = video_id_to_frames[video_id] # (a list of two frame numbers, one for edv and one for esv)
        lower_frame = frames[0]
        higher_frame = frames[1]

        video_cap = cv2.VideoCapture(video_path)
        success, img = video_cap.read() # (img has shape: (112, 112, 3))
        frame_count = 0
        while success:
            if frame_count == higher_frame:
                imgs_train.append(img)
                labels_train.append(esv)
                # cv2.imwrite("/root/regression_uncertainty/VentricularVolume/%s_frame_%d_volume_%f.png" % (video_id, frame_count, esv), img)

                break
            else:
                success, img = video_cap.read() # (img has shape: (112, 112, 3))
                frame_count += 1

num_imgs_train = len(imgs_train)
print ("num imgs train: %d" % num_imgs_train)


imgs_val = []
labels_val = []
for i in range(num_videos_val):
    if i % 100 == 0:
        print (i)

    video_id = video_ids_val[i]
    esv = esvs_val[i] # (smaller volume)
    edv = edvs_val[i] # (large volume)

    video_path = echonet_dynamic_data_path + "EchoNet-Dynamic/Videos/" + video_id + ".avi"

    if video_id in video_id_to_frames: # (a small number of videos apparently don't have tracings)
        frames = video_id_to_frames[video_id] # (a list of two frame numbers, one for edv and one for esv)
        lower_frame = frames[0]
        higher_frame = frames[1]

        video_cap = cv2.VideoCapture(video_path)
        success, img = video_cap.read() # (img has shape: (112, 112, 3))
        frame_count = 0
        while success:
            if frame_count == higher_frame:
                imgs_val.append(img)
                labels_val.append(esv)
                # cv2.imwrite("/root/regression_uncertainty/VentricularVolume/%s_frame_%d_volume_%f.png" % (video_id, frame_count, esv), img)

                break
            else:
                success, img = video_cap.read() # (img has shape: (112, 112, 3))
                frame_count += 1

num_imgs_val = len(imgs_val)
print ("num imgs val: %d" % num_imgs_val)


imgs_test = []
labels_test = []
for i in range(num_videos_test):
    if i % 100 == 0:
        print (i)

    video_id = video_ids_test[i]
    esv = esvs_test[i] # (smaller volume)
    edv = edvs_test[i] # (large volume)

    video_path = echonet_dynamic_data_path + "EchoNet-Dynamic/Videos/" + video_id + ".avi"

    if video_id in video_id_to_frames: # (a small number of videos apparently don't have tracings)
        frames = video_id_to_frames[video_id] # (a list of two frame numbers, one for edv and one for esv)
        lower_frame = frames[0]
        higher_frame = frames[1]

        video_cap = cv2.VideoCapture(video_path)
        success, img = video_cap.read() # (img has shape: (112, 112, 3))
        frame_count = 0
        while success:
            if frame_count == lower_frame:
                imgs_test.append(img)
                labels_test.append(edv)
                # cv2.imwrite("/root/regression_uncertainty/VentricularVolume/%s_frame_%d_volume_%f.png" % (video_id, frame_count, edv), img)

                break
            else:
                success, img = video_cap.read() # (img has shape: (112, 112, 3))
                frame_count += 1

num_imgs_test = len(imgs_test)
print ("num imgs test: %d" % num_imgs_test)




images_train = np.zeros((num_imgs_train, 64, 64, 3), dtype=np.uint8)
for i in range(num_imgs_train):
    img = imgs_train[i] # (shape: (112, 112, 3))
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
    images_train[i] = img
labels_train = np.array(labels_train).astype(np.float32)

images_val = np.zeros((num_imgs_val, 64, 64, 3), dtype=np.uint8)
for i in range(num_imgs_val):
    img = imgs_val[i] # (shape: (112, 112, 3))
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
    images_val[i] = img
labels_val = np.array(labels_val).astype(np.float32)

images_test = np.zeros((num_imgs_test, 64, 64, 3), dtype=np.uint8)
for i in range(num_imgs_test):
    img = imgs_test[i] # (shape: (112, 112, 3))
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR) # (shape: (64, 64, 3))
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
#
print (np.min(labels_test))
print (np.max(labels_test))
print (np.mean(labels_test))

with open("/root/regression_uncertainty/VentricularVolume/labels_train.pkl", "wb") as file:
    pickle.dump(labels_train, file)
with open("/root/regression_uncertainty/VentricularVolume/images_train.pkl", "wb") as file:
    pickle.dump(images_train, file)

with open("/root/regression_uncertainty/VentricularVolume/labels_val.pkl", "wb") as file:
    pickle.dump(labels_val, file)
with open("/root/regression_uncertainty/VentricularVolume/images_val.pkl", "wb") as file:
    pickle.dump(images_val, file)

with open("/root/regression_uncertainty/VentricularVolume/labels_test.pkl", "wb") as file:
    pickle.dump(labels_test, file)
with open("/root/regression_uncertainty/VentricularVolume/images_test.pkl", "wb") as file:
    pickle.dump(images_test, file)
################################################################################
