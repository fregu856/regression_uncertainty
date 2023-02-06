# camera-ready

from datasets import DatasetTrain, DatasetVal, DatasetTest # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model_gaussian import GaussNet

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from sklearn import mixture

import torch.distributions
from scipy.special import logsumexp

import scipy.stats

model_id = "direct_regression"
num_models = 20


alpha = 0.1


M = 1

np.random.seed(0)

epoch = 75

batch_size = 32

train_dataset = DatasetTrain()
val_dataset = DatasetVal()
test_dataset = DatasetTest()

num_val_batches = int(len(val_dataset)/batch_size)
print ("num_val_batches:", num_val_batches)

num_test_batches = int(len(test_dataset)/batch_size)
print ("num_test_batches:", num_test_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model_inds = list(range(num_models))

num_runs = 5


val_maes = []
val_coverages = []
val_avg_lengths = []
test_maes = []
test_coverages = []
test_avg_lengths = []
for run_i in range(num_runs):
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)

    networks = []
    for i in range(M):
        model_i = model_inds[i]
        network = GaussNet(model_id + "_eval_conformal_prediction", project_dir="/root/regression_uncertainty/VentricularVolume").cuda()
        network.load_state_dict(torch.load("/root/regression_uncertainty/VentricularVolume/training_logs/model_%s_%d/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))
        networks.append(network)

    M_float = float(len(networks))
    print (M_float)




    ############################################################################
    pred_values = []
    y_values = []
    abs_error_values = []
    for network in networks:
        network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(val_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size))

            preds = []
            for network in networks:
                x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
                pred, _ = network.head_net(x_features) # (shape: (batch_size, 1))
                pred = pred.view(-1) # (shape: (batch_size))
                preds.append(pred)

            pred = torch.zeros(preds[0].size()).cuda()
            for value in preds:
                pred = pred + value/M_float
            preds = pred

            abs_error = torch.abs(preds - ys)

            preds = preds.data.cpu().numpy() # (shape: (batch_size, ))
            pred_values += list(preds)

            ys = ys.data.cpu().numpy() # (shape: (batch_size, ))
            y_values += list(ys)

            abs_error = abs_error.data.cpu().numpy() # (shape: (batch_size, ))
            abs_error_values += list(abs_error)

    pred_values = np.array(pred_values)
    y_values = np.array(y_values)
    abs_error_values = np.array(abs_error_values)

    num_predictions = float(y_values.shape[0])

    conformity_scores = abs_error_values # (shape: (num_predictons, ))
    sorted_conformity_scores = np.sort(conformity_scores) # (the first element is the smallest value)

    Q_1_alpha = sorted_conformity_scores[int((1.0 - alpha)*num_predictions)]

    lower_values = pred_values - Q_1_alpha
    upper_values = pred_values + Q_1_alpha

    val_coverage = np.count_nonzero(np.logical_and(y_values >= lower_values, y_values <= upper_values))/num_predictions
    val_coverages.append(val_coverage)

    val_avg_length = np.mean(upper_values - lower_values)
    val_avg_lengths.append(val_avg_length)

    val_mae = np.mean(abs_error_values)
    val_maes.append(val_mae)




    ############################################################################
    pred_values = []
    y_values = []
    abs_error_values = []
    for network in networks:
        network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size))

            preds = []
            for network in networks:
                x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
                pred, _ = network.head_net(x_features) # (shape: (batch_size, 1))
                pred = pred.view(-1) # (shape: (batch_size))
                preds.append(pred)

            pred = torch.zeros(preds[0].size()).cuda()
            for value in preds:
                pred = pred + value/M_float
            preds = pred

            abs_error = torch.abs(preds - ys)

            preds = preds.data.cpu().numpy() # (shape: (batch_size, ))
            pred_values += list(preds)

            ys = ys.data.cpu().numpy() # (shape: (batch_size, ))
            y_values += list(ys)

            abs_error = abs_error.data.cpu().numpy() # (shape: (batch_size, ))
            abs_error_values += list(abs_error)

    pred_values = np.array(pred_values)
    y_values = np.array(y_values)
    abs_error_values = np.array(abs_error_values)

    num_predictions = float(y_values.shape[0])

    lower_values = pred_values - Q_1_alpha
    upper_values = pred_values + Q_1_alpha

    test_coverage = np.count_nonzero(np.logical_and(y_values >= lower_values, y_values <= upper_values))/num_predictions
    test_coverages.append(test_coverage)

    test_avg_length = np.mean(upper_values - lower_values)
    test_avg_lengths.append(test_avg_length)

    test_mae = np.mean(abs_error_values)
    test_maes.append(test_mae)




    print ("val_maes:")
    print (val_maes)
    print ("val_maes: %g +/- %g" % (np.mean(np.array(val_maes)), np.std(np.array(val_maes))))
    print ("$")
    print ("$")
    print ("val_coverages:")
    print (val_coverages)
    print ("val_coverages: %g +/- %g" % (np.mean(np.array(val_coverages)), np.std(np.array(val_coverages))))
    print ("$")
    print ("$")
    print ("val_avg_lengths:")
    print (val_avg_lengths)
    print ("val_avg_lengths: %g +/- %g" % (np.mean(np.array(val_avg_lengths)), np.std(np.array(val_avg_lengths))))
    print ("$")
    print ("$")
    print ("test_maes:")
    print (test_maes)
    print ("test_maes: %g +/- %g" % (np.mean(np.array(test_maes)), np.std(np.array(test_maes))))
    print ("$")
    print ("$")
    print ("test_coverages:")
    print (test_coverages)
    print ("test_coverages: %g +/- %g" % (np.mean(np.array(test_coverages)), np.std(np.array(test_coverages))))
    print ("$")
    print ("$")
    print ("test_avg_lengths:")
    print (test_avg_lengths)
    print ("test_avg_lengths: %g +/- %g" % (np.mean(np.array(test_avg_lengths)), np.std(np.array(test_avg_lengths))))

    print ("###################################################################")
    print ("###################################################################")
    print ("###################################################################")
    print ("###################################################################")
