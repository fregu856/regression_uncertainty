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

model_id = "gaussian"
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
val_coverages_before_calib = []
val_avg_lengths_before_calib = []
val_coverages = []
val_avg_lengths = []
test_maes = []
test_coverages = []
test_avg_lengths = []
#
sigma_alea_test_pred_props = []
sigma_alea_test_coverages = []
sigma_alea_test_avg_lengths = []
#
sigma_epi_test_pred_props = []
sigma_epi_test_coverages = []
sigma_epi_test_avg_lengths = []
#
sigma_pred_test_pred_props = []
sigma_pred_test_coverages = []
sigma_pred_test_avg_lengths = []
for run_i in range(num_runs):
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)

    networks = []
    for i in range(M):
        model_i = model_inds[i]
        network = GaussNet(model_id + "_eval_gaussian_selective_variance", project_dir="/root/regression_uncertainty/SkinLesionPixels").cuda()
        network.load_state_dict(torch.load("/root/regression_uncertainty/SkinLesionPixels/training_logs/model_%s_%d/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))
        networks.append(network)

    M_float = float(len(networks))
    print (M_float)




    ############################################################################
    mean_values = []
    sigma_values = []
    sigma_alea_values = []
    sigma_epi_values = []
    y_values = []
    abs_error_values = []
    for network in networks:
        network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(val_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size))

            means = []
            sigma2s = []
            for network in networks:
                x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
                mean, log_sigma2 = network.head_net(x_features) # (both has shape: (batch_size, 1))
                mean = mean.view(-1) # (shape: (batch_size))
                log_sigma2 = log_sigma2.view(-1) # (shape: (batch_size))
                sigma2 = torch.exp(log_sigma2)
                means.append(mean)
                sigma2s.append(sigma2)

            mean = torch.zeros(means[0].size()).cuda()
            for value in means:
                mean = mean + value/M_float

            sigma_2_alea = torch.zeros(means[0].size()).cuda()
            for value in sigma2s:
                sigma_2_alea = sigma_2_alea + value/M_float

            sigma_2_epi = torch.zeros(means[0].size()).cuda()
            for value in means:
                sigma_2_epi = sigma_2_epi + torch.pow(mean - value, 2)/M_float

            sigma_2_pred = sigma_2_alea + sigma_2_epi

            means = mean
            sigmas = torch.sqrt(sigma_2_pred)
            sigmas_alea = torch.sqrt(sigma_2_alea)
            sigmas_epi = torch.sqrt(sigma_2_epi)

            abs_error = torch.abs(means - ys)

            means = means.data.cpu().numpy() # (shape: (batch_size, ))
            mean_values += list(means)

            sigmas = sigmas.data.cpu().numpy() # (shape: (batch_size, ))
            sigma_values += list(sigmas)

            sigmas_alea = sigmas_alea.data.cpu().numpy() # (shape: (batch_size, ))
            sigma_alea_values += list(sigmas_alea)

            sigmas_epi = sigmas_epi.data.cpu().numpy() # (shape: (batch_size, ))
            sigma_epi_values += list(sigmas_epi)

            ys = ys.data.cpu().numpy() # (shape: (batch_size, ))
            y_values += list(ys)

            abs_error = abs_error.data.cpu().numpy() # (shape: (batch_size, ))
            abs_error_values += list(abs_error)

    mean_values = np.array(mean_values)
    sigma_values = np.array(sigma_values)
    y_values = np.array(y_values)
    abs_error_values = np.array(abs_error_values)

    sigma_alea_values = np.array(sigma_alea_values)
    sigma_epi_values = np.array(sigma_epi_values)
    sigma_pred_values = np.copy(sigma_values)

    num_preds = abs_error_values.shape[0]
    #
    sorted_inds_sigma_alea = np.argsort(sigma_alea_values)
    threshold_value_sigma_alea = sigma_alea_values[sorted_inds_sigma_alea[int((0.95)*num_preds)]]
    #
    sorted_inds_sigma_epi = np.argsort(sigma_epi_values)
    threshold_value_sigma_epi = sigma_epi_values[sorted_inds_sigma_epi[int((0.95)*num_preds)]]
    #
    sorted_inds_sigma_pred = np.argsort(sigma_pred_values)
    threshold_value_sigma_pred = sigma_pred_values[sorted_inds_sigma_pred[int((0.95)*num_preds)]]

    val_mae = np.mean(abs_error_values)
    val_maes.append(val_mae)

    num_predictions = float(y_values.shape[0])

    lower_values = mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_values
    upper_values = mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_values

    val_coverage_before_calib = np.count_nonzero(np.logical_and(y_values >= lower_values, y_values <= upper_values))/num_predictions
    val_coverages_before_calib.append(val_coverage_before_calib)

    val_avg_length_before_calib = np.mean(upper_values - lower_values)
    val_avg_lengths_before_calib.append(val_avg_length_before_calib)


    conformity_scores = np.maximum(lower_values - y_values, y_values - upper_values) # (shape: (num_predictons, ))
    sorted_conformity_scores = np.sort(conformity_scores) # (the first element is the smallest value)

    Q_1_alpha = sorted_conformity_scores[int((1.0 - alpha)*num_predictions)]

    lower_values = lower_values - Q_1_alpha
    upper_values = upper_values + Q_1_alpha

    val_coverage = np.count_nonzero(np.logical_and(y_values >= lower_values, y_values <= upper_values))/num_predictions
    val_coverages.append(val_coverage)

    val_avg_length = np.mean(upper_values - lower_values)
    val_avg_lengths.append(val_avg_length)



    ############################################################################
    mean_values = []
    sigma_values = []
    sigma_alea_values = []
    sigma_epi_values = []
    y_values = []
    abs_error_values = []
    for network in networks:
        network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size))

            means = []
            sigma2s = []
            for network in networks:
                x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
                mean, log_sigma2 = network.head_net(x_features) # (both has shape: (batch_size, 1))
                mean = mean.view(-1) # (shape: (batch_size))
                log_sigma2 = log_sigma2.view(-1) # (shape: (batch_size))
                sigma2 = torch.exp(log_sigma2)
                means.append(mean)
                sigma2s.append(sigma2)

            mean = torch.zeros(means[0].size()).cuda()
            for value in means:
                mean = mean + value/M_float

            sigma_2_alea = torch.zeros(means[0].size()).cuda()
            for value in sigma2s:
                sigma_2_alea = sigma_2_alea + value/M_float

            sigma_2_epi = torch.zeros(means[0].size()).cuda()
            for value in means:
                sigma_2_epi = sigma_2_epi + torch.pow(mean - value, 2)/M_float

            sigma_2_pred = sigma_2_alea + sigma_2_epi

            means = mean
            sigmas = torch.sqrt(sigma_2_pred)
            sigmas_alea = torch.sqrt(sigma_2_alea)
            sigmas_epi = torch.sqrt(sigma_2_epi)

            abs_error = torch.abs(means - ys)

            means = means.data.cpu().numpy() # (shape: (batch_size, ))
            mean_values += list(means)

            sigmas = sigmas.data.cpu().numpy() # (shape: (batch_size, ))
            sigma_values += list(sigmas)

            sigmas_alea = sigmas_alea.data.cpu().numpy() # (shape: (batch_size, ))
            sigma_alea_values += list(sigmas_alea)

            sigmas_epi = sigmas_epi.data.cpu().numpy() # (shape: (batch_size, ))
            sigma_epi_values += list(sigmas_epi)

            ys = ys.data.cpu().numpy() # (shape: (batch_size, ))
            y_values += list(ys)

            abs_error = abs_error.data.cpu().numpy() # (shape: (batch_size, ))
            abs_error_values += list(abs_error)

    mean_values = np.array(mean_values)
    sigma_values = np.array(sigma_values)
    y_values = np.array(y_values)
    abs_error_values = np.array(abs_error_values)

    sigma_alea_values = np.array(sigma_alea_values)
    sigma_epi_values = np.array(sigma_epi_values)
    sigma_pred_values = np.copy(sigma_values)

    test_mae = np.mean(abs_error_values)
    test_maes.append(test_mae)

    num_predictions = float(y_values.shape[0])

    lower_values = mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_values
    upper_values = mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_values

    lower_values = lower_values - Q_1_alpha ###########################################################################################
    upper_values = upper_values + Q_1_alpha ###########################################################################################

    test_coverage = np.count_nonzero(np.logical_and(y_values >= lower_values, y_values <= upper_values))/float(y_values.shape[0])
    test_coverages.append(test_coverage)

    test_avg_length = np.mean(upper_values - lower_values)
    test_avg_lengths.append(test_avg_length)




    sigma_alea_sigma_values = sigma_values[sigma_alea_values <= threshold_value_sigma_alea]
    sigma_alea_mean_values = mean_values[sigma_alea_values <= threshold_value_sigma_alea]
    sigma_alea_y_values = y_values[sigma_alea_values <= threshold_value_sigma_alea]
    #
    sigma_alea_pred_prop = float(sigma_alea_y_values.shape[0])/float(y_values.shape[0])
    sigma_alea_test_pred_props.append(sigma_alea_pred_prop)
    #
    sigma_alea_lower_values = sigma_alea_mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_alea_sigma_values
    sigma_alea_upper_values = sigma_alea_mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_alea_sigma_values
    #
    sigma_alea_lower_values = sigma_alea_lower_values - Q_1_alpha ###########################################################################################
    sigma_alea_upper_values = sigma_alea_upper_values + Q_1_alpha ###########################################################################################
    #
    sigma_alea_test_coverage = np.count_nonzero(np.logical_and(sigma_alea_y_values >= sigma_alea_lower_values, sigma_alea_y_values <= sigma_alea_upper_values))/float(sigma_alea_y_values.shape[0])
    sigma_alea_test_coverages.append(sigma_alea_test_coverage)
    #
    sigma_alea_test_avg_length = np.mean(sigma_alea_upper_values - sigma_alea_lower_values)
    sigma_alea_test_avg_lengths.append(sigma_alea_test_avg_length)


    sigma_epi_sigma_values = sigma_values[sigma_epi_values <= threshold_value_sigma_epi]
    sigma_epi_mean_values = mean_values[sigma_epi_values <= threshold_value_sigma_epi]
    sigma_epi_y_values = y_values[sigma_epi_values <= threshold_value_sigma_epi]
    #
    sigma_epi_pred_prop = float(sigma_epi_y_values.shape[0])/float(y_values.shape[0])
    sigma_epi_test_pred_props.append(sigma_epi_pred_prop)
    #
    sigma_epi_lower_values = sigma_epi_mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_epi_sigma_values
    sigma_epi_upper_values = sigma_epi_mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_epi_sigma_values
    #
    sigma_epi_lower_values = sigma_epi_lower_values - Q_1_alpha ###########################################################################################
    sigma_epi_upper_values = sigma_epi_upper_values + Q_1_alpha ###########################################################################################
    #
    sigma_epi_test_coverage = np.count_nonzero(np.logical_and(sigma_epi_y_values >= sigma_epi_lower_values, sigma_epi_y_values <= sigma_epi_upper_values))/float(sigma_epi_y_values.shape[0])
    sigma_epi_test_coverages.append(sigma_epi_test_coverage)
    #
    sigma_epi_test_avg_length = np.mean(sigma_epi_upper_values - sigma_epi_lower_values)
    sigma_epi_test_avg_lengths.append(sigma_epi_test_avg_length)


    sigma_pred_sigma_values = sigma_values[sigma_pred_values <= threshold_value_sigma_pred]
    sigma_pred_mean_values = mean_values[sigma_pred_values <= threshold_value_sigma_pred]
    sigma_pred_y_values = y_values[sigma_pred_values <= threshold_value_sigma_pred]
    #
    sigma_pred_pred_prop = float(sigma_pred_y_values.shape[0])/float(y_values.shape[0])
    sigma_pred_test_pred_props.append(sigma_pred_pred_prop)
    #
    sigma_pred_lower_values = sigma_pred_mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_pred_sigma_values
    sigma_pred_upper_values = sigma_pred_mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*sigma_pred_sigma_values
    #
    sigma_pred_lower_values = sigma_pred_lower_values - Q_1_alpha ###########################################################################################
    sigma_pred_upper_values = sigma_pred_upper_values + Q_1_alpha ###########################################################################################
    #
    sigma_pred_test_coverage = np.count_nonzero(np.logical_and(sigma_pred_y_values >= sigma_pred_lower_values, sigma_pred_y_values <= sigma_pred_upper_values))/float(sigma_pred_y_values.shape[0])
    sigma_pred_test_coverages.append(sigma_pred_test_coverage)
    #
    sigma_pred_test_avg_length = np.mean(sigma_pred_upper_values - sigma_pred_lower_values)
    sigma_pred_test_avg_lengths.append(sigma_pred_test_avg_length)




    print ("val_maes:")
    print (val_maes)
    print ("val_maes: %g +/- %g" % (np.mean(np.array(val_maes)), np.std(np.array(val_maes))))
    print ("$")
    print ("$")
    print ("val_coverages_before_calib:")
    print (val_coverages_before_calib)
    print ("val_coverages_before_calib: %g +/- %g" % (np.mean(np.array(val_coverages_before_calib)), np.std(np.array(val_coverages_before_calib))))
    print ("$")
    print ("$")
    print ("val_avg_lengths_before_calib:")
    print (val_avg_lengths_before_calib)
    print ("val_avg_lengths_before_calib: %g +/- %g" % (np.mean(np.array(val_avg_lengths_before_calib)), np.std(np.array(val_avg_lengths_before_calib))))
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
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    print ("sigma_alea_test_pred_props:")
    print (sigma_alea_test_pred_props)
    print ("sigma_alea_test_pred_props: %g +/- %g" % (np.mean(np.array(sigma_alea_test_pred_props)), np.std(np.array(sigma_alea_test_pred_props))))
    print ("$")
    print ("$")
    print ("sigma_alea_test_coverages:")
    print (sigma_alea_test_coverages)
    print ("sigma_alea_test_coverages: %g +/- %g" % (np.mean(np.array(sigma_alea_test_coverages)), np.std(np.array(sigma_alea_test_coverages))))
    print ("$")
    print ("$")
    print ("sigma_alea_test_avg_lengths:")
    print (sigma_alea_test_avg_lengths)
    print ("sigma_alea_test_avg_lengths: %g +/- %g" % (np.mean(np.array(sigma_alea_test_avg_lengths)), np.std(np.array(sigma_alea_test_avg_lengths))))
    print ("###################################################################")
    print ("###################################################################")
    print ("###################################################################")
    print ("###################################################################")
