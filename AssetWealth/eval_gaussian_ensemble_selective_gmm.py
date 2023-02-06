# camera-ready

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


M = 5

np.random.seed(0)

epoch = 75

batch_size = 32

################################################################################
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms

dataset = get_dataset(dataset="poverty", download=False)
train_dataset = dataset.get_subset("train", transform=transforms.Resize((64, 64)))
val_dataset = dataset.get_subset("id_val", transform=transforms.Resize((64, 64)))
test_dataset = dataset.get_subset("test", transform=transforms.Resize((64, 64)))

num_train_batches = int(len(train_dataset)/batch_size)
print (len(train_dataset))
print ("num_train_batches:", num_train_batches)

num_val_batches = int(len(val_dataset)/batch_size)
print (len(val_dataset))
print ("num_val_batches:", num_val_batches)

num_test_batches = int(len(test_dataset)/batch_size)
print (len(test_dataset))
print ("num_test_batches:", num_test_batches)

train_loader = get_train_loader("standard", train_dataset, batch_size=batch_size)
val_loader = get_eval_loader("standard", val_dataset, batch_size=batch_size)
test_loader = get_eval_loader("standard", test_dataset, batch_size=batch_size)
################################################################################

model_inds = list(range(num_models))

num_runs = 5




################################################################################
# (this is just a fix to evaluate the same models as in eval_gaussian_ensemble.py, and thus get the same val_maes etc (I guess that the GaussianMixture invokes the np.random somehow, changing which models are selected after the shuffeling))
model_is_list = []
for run_i in range(num_runs):
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)

    model_is = []
    for i in range(M):
        model_is.append(model_inds[i])
    model_is_list.append(model_is)
print(model_is_list)
################################################################################




val_maes = []
val_coverages_before_calib = []
val_avg_lengths_before_calib = []
val_coverages = []
val_avg_lengths = []
test_maes = []
test_coverages = []
test_avg_lengths = []
#
gmm_prob_test_pred_props = []
gmm_prob_test_coverages = []
gmm_prob_test_avg_lengths = []
for run_i in range(num_runs):
    networks = []
    for i in range(M):
        model_i = model_is_list[run_i][i]
        network = GaussNet(model_id + "_eval_gaussian_ensemble_selective_gmm", project_dir="/root/regression_uncertainty/AssetWealth").cuda()
        network.load_state_dict(torch.load("/root/regression_uncertainty/AssetWealth/training_logs/model_%s_%d/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))
        networks.append(network)

    M_float = float(len(networks))
    print (M_float)




    ############################################################################
    feature_gmms = []
    for network in networks:
        network.eval() # (set in eval mode, this affects BatchNorm and dropout)
        for step, (xs, ys, _) in enumerate(train_loader):
            with torch.no_grad():
                xs = xs.cuda() # (shape: (batch_size, 8, img_size, img_size))

                x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))

                if step == 0:
                    features = x_features
                else:
                    features = torch.cat([features, x_features], 0)

        # (features has shape: (num_train_examples, hidden_dim))

        feature_gmm = mixture.GaussianMixture(n_components=4, covariance_type="full")
        feature_gmm.fit(features.cpu().numpy())
        feature_gmms.append(feature_gmm)
    ############################################################################




    ############################################################################
    mean_values = []
    sigma_values = []
    y_values = []
    abs_error_values = []
    gmm_prob_values = []
    for network in networks:
        network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys, _) in enumerate(val_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 8, img_size, img_size))
            ys = ys.cuda().squeeze(1) # (shape: (batch_size))

            means = []
            sigma2s = []
            features_list = []
            for network in networks:
                features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
                mean, log_sigma2 = network.head_net(features) # (both has shape: (batch_size, 1))
                mean = mean.view(-1) # (shape: (batch_size))
                log_sigma2 = log_sigma2.view(-1) # (shape: (batch_size))
                sigma2 = torch.exp(log_sigma2)
                #
                means.append(mean)
                sigma2s.append(sigma2)
                features_list.append(features)

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

            gmm_log_probs = np.zeros((means.size(0), M)) # (shape: (batch_size, M))
            for i in range(len(features_list)):
                gmm_log_prob = feature_gmms[i].score_samples(features_list[i].cpu().numpy()) # (shape: (batch_size, ))
                gmm_log_probs[:, i] = gmm_log_prob
            gmm_log_prob = logsumexp(gmm_log_probs, axis=1) - np.log(M_float) # (shape: (batch_size, ))
            gmm_probs = gmm_log_prob # (this is actually the log probability, not the probability)
            gmm_prob_values += list(gmm_probs)

            abs_error = torch.abs(means - ys)

            means = means.data.cpu().numpy() # (shape: (batch_size, ))
            mean_values += list(means)

            sigmas = sigmas.data.cpu().numpy() # (shape: (batch_size, ))
            sigma_values += list(sigmas)

            ys = ys.data.cpu().numpy() # (shape: (batch_size, ))
            y_values += list(ys)

            abs_error = abs_error.data.cpu().numpy() # (shape: (batch_size, ))
            abs_error_values += list(abs_error)

    mean_values = np.array(mean_values)
    sigma_values = np.array(sigma_values)
    y_values = np.array(y_values)
    abs_error_values = np.array(abs_error_values)

    gmm_prob_values = np.array(gmm_prob_values)

    num_preds = abs_error_values.shape[0]
    sorted_inds_gmm = np.argsort(list(-np.array(gmm_prob_values)))
    threshold_value_gmm_prob = gmm_prob_values[sorted_inds_gmm[int((0.95)*num_preds)]]

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
    y_values = []
    abs_error_values = []
    gmm_prob_values = []
    for network in networks:
        network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys, _) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 8, img_size, img_size))
            ys = ys.cuda().squeeze(1) # (shape: (batch_size))

            means = []
            sigma2s = []
            features_list = []
            for network in networks:
                features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
                mean, log_sigma2 = network.head_net(features) # (both has shape: (batch_size, 1))
                mean = mean.view(-1) # (shape: (batch_size))
                log_sigma2 = log_sigma2.view(-1) # (shape: (batch_size))
                sigma2 = torch.exp(log_sigma2)
                #
                means.append(mean)
                sigma2s.append(sigma2)
                features_list.append(features)

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

            gmm_log_probs = np.zeros((means.size(0), M)) # (shape: (batch_size, M))
            for i in range(len(features_list)):
                gmm_log_prob = feature_gmms[i].score_samples(features_list[i].cpu().numpy()) # (shape: (batch_size, ))
                gmm_log_probs[:, i] = gmm_log_prob
            gmm_log_prob = logsumexp(gmm_log_probs, axis=1) - np.log(M_float) # (shape: (batch_size, ))
            gmm_probs = gmm_log_prob # (this is actually the log probability, not the probability)
            gmm_prob_values += list(gmm_probs)

            abs_error = torch.abs(means - ys)

            means = means.data.cpu().numpy() # (shape: (batch_size, ))
            mean_values += list(means)

            sigmas = sigmas.data.cpu().numpy() # (shape: (batch_size, ))
            sigma_values += list(sigmas)

            ys = ys.data.cpu().numpy() # (shape: (batch_size, ))
            y_values += list(ys)

            abs_error = abs_error.data.cpu().numpy() # (shape: (batch_size, ))
            abs_error_values += list(abs_error)

    mean_values = np.array(mean_values)
    sigma_values = np.array(sigma_values)
    y_values = np.array(y_values)
    abs_error_values = np.array(abs_error_values)

    gmm_prob_values = np.array(gmm_prob_values)

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


    gmm_prob_sigma_values = sigma_values[gmm_prob_values >= threshold_value_gmm_prob]
    gmm_prob_mean_values = mean_values[gmm_prob_values >= threshold_value_gmm_prob]
    gmm_prob_y_values = y_values[gmm_prob_values >= threshold_value_gmm_prob]
    #
    gmm_prob_pred_prop = float(gmm_prob_y_values.shape[0])/float(y_values.shape[0])
    gmm_prob_test_pred_props.append(gmm_prob_pred_prop)
    #
    gmm_prob_lower_values = gmm_prob_mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*gmm_prob_sigma_values
    gmm_prob_upper_values = gmm_prob_mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*gmm_prob_sigma_values
    #
    gmm_prob_lower_values = gmm_prob_lower_values - Q_1_alpha ###########################################################################################
    gmm_prob_upper_values = gmm_prob_upper_values + Q_1_alpha ###########################################################################################
    #
    gmm_prob_test_coverage = np.count_nonzero(np.logical_and(gmm_prob_y_values >= gmm_prob_lower_values, gmm_prob_y_values <= gmm_prob_upper_values))/float(gmm_prob_y_values.shape[0])
    gmm_prob_test_coverages.append(gmm_prob_test_coverage)
    #
    gmm_prob_test_avg_length = np.mean(gmm_prob_upper_values - gmm_prob_lower_values)
    gmm_prob_test_avg_lengths.append(gmm_prob_test_avg_length)


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

    print ("gmm_prob_test_pred_props:")
    print (gmm_prob_test_pred_props)
    print ("gmm_prob_test_pred_props: %g +/- %g" % (np.mean(np.array(gmm_prob_test_pred_props)), np.std(np.array(gmm_prob_test_pred_props))))
    print ("$")
    print ("$")
    print ("gmm_prob_test_coverages:")
    print (gmm_prob_test_coverages)
    print ("gmm_prob_test_coverages: %g +/- %g" % (np.mean(np.array(gmm_prob_test_coverages)), np.std(np.array(gmm_prob_test_coverages))))
    print ("$")
    print ("$")
    print ("gmm_prob_test_avg_lengths:")
    print (gmm_prob_test_avg_lengths)
    print ("gmm_prob_test_avg_lengths: %g +/- %g" % (np.mean(np.array(gmm_prob_test_avg_lengths)), np.std(np.array(gmm_prob_test_avg_lengths))))
    print ("###################################################################")
    print ("###################################################################")
    print ("###################################################################")
    print ("###################################################################")
