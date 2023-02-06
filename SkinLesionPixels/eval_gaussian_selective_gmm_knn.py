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




################################################################################
from annoy import AnnoyIndex
from typing import Tuple

class ApproxNearestNeighbors:
    """
    Approximate nearest neighbor search is done with Annoy using binary trees. This is blazing fast.
    Reference:
    https://erikbern.com/2015/10/01/nearest-neighbors-and-vector-models-part-2-how-to-search-in-high-dimensional-spaces.html
    Parameters
    ----------
    features: Array of shape (n_samples, n_features)
        Features for each observation.
    labels: Array of shape (n_samples,)
        Given class label indices.
    features_key: Array of shape (n_samples,)
        Optional unique key for each observation (e.g. image paths for images)
    """

    def __init__(
        self, features: np.array, labels: np.array = None, features_key: np.array = None
    ):

        self.ann_index = None  # init empty approximate nearest neighbor index
        self.features = features
        self.labels = labels
        self.features_key = features_key
        self.num_labels = np.unique(labels).shape[0]  # number of unique labels
        self.num_obs = features.shape[0]  # number of observations

        # init nearest neighbors
        self.neighbors_idx = None
        self.neighbors_dist = None
        self.neighbors_labels = None

        # init predicted probabilities
        self.pred_probs = None

    def build_index(self, metric: str = "angular", n_trees: int = 10):
        """
        Build approximate nearest neighbors index.
        Parameters
        ----------
        metric: {"angular", "euclidean", "manhattan", "hamming", "dot"}, default="angular"
            Distance metric for approximate nearest neighbor search.
        n_trees: int, default=10
            Number of trees to use for building the index. More trees will give more accurate results but larger indexes.
            Annoy uses binary trees. See Annoy documentation for more details.
            https://github.com/spotify/annoy
        """

        # dimension of feature space
        dim = self.features.shape[1]

        # build approximate nearest neighbor index
        print("Building nearest neighbors index")
        self.ann_index = AnnoyIndex(dim, metric)
        for i, x in enumerate(self.features):
            self.ann_index.add_item(i, x)
        self.ann_index.build(n_trees)

        return self.ann_index

    def get_k_nearest_neighbors(self, k: int = 25) -> Tuple[np.array]:
        """
        Get the k nearest neighbors for each observation.
        Parameters
        ----------
        k: int, default=25
            Number of nearest neighbors to search for.
        Returns
        -------
        k_nearest_neighbors: Tuple of arrays of shape (n_samples, n_neighbors)
            First array contains the indices for the nearest neighbors.
            Second array contains the distances to the nearest neighbors.
            Third array contains the labels for the nearest neighbors.
            Fourth array contains the weights for the nearest neighbors.
        """

        # get k nearest neighbors for each observation
        neighbors_idx = []
        neighbors_dist = []
        for i in range(self.features.shape[0]):

            # note: we need to do k + 1 because the first neighbor for each observation is itself
            idx, dist = self.ann_index.get_nns_by_item(i, k + 1, include_distances=True)

            # save neighbors to list
            # note: we exclude the first neighbor because the first neighbor for each observation is itself
            neighbors_idx.append(idx[1:])
            neighbors_dist.append(dist[1:])

        # convert to numpy array
        self.neighbors_idx = np.array(neighbors_idx)
        self.neighbors_dist = np.array(neighbors_dist)

        return (
            self.neighbors_idx,
            self.neighbors_dist,
        )
################################################################################


################################################################################
# (this is just a fix to evaluate the same num_runs models as in eval_gaussian.py, and thus get the same val_maes etc (I guess that the ApproxNearestNeighbors invokes the np.random somehow, changing which models are selected after the shuffeling))
model_is = []
for run_i in range(num_runs):
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    np.random.shuffle(model_inds)
    model_is.append(model_inds[0])
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
#
knndist_test_pred_props = []
knndist_test_coverages = []
knndist_test_avg_lengths = []
#
knndist_l2_test_pred_props = []
knndist_l2_test_coverages = []
knndist_l2_test_avg_lengths = []
for run_i in range(num_runs):
    model_i = model_is[run_i]
    network = GaussNet(model_id + "_eval_gaussian_selective_gmm_knn", project_dir="/root/regression_uncertainty/SkinLesionPixels").cuda()
    network.load_state_dict(torch.load("/root/regression_uncertainty/SkinLesionPixels/training_logs/model_%s_%d/checkpoints/model_%s_epoch_%d.pth" % (model_id, model_i, model_id, epoch)))




    ############################################################################
    network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(train_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))

            x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))

            if step == 0:
                features = x_features
            else:
                features = torch.cat([features, x_features], 0)

    # (features has shape: (num_train_examples, hidden_dim))

    feature_gmm = mixture.GaussianMixture(n_components=4, covariance_type="full")
    feature_gmm.fit(features.cpu().numpy())

    nns = ApproxNearestNeighbors(features=features.cpu().numpy())
    nns.build_index()
    #
    nns_l2 = ApproxNearestNeighbors(features=features.cpu().numpy())
    nns_l2.build_index(metric="euclidean")
    ############################################################################




    ############################################################################
    mean_values = []
    sigma_values = []
    y_values = []
    abs_error_values = []
    gmm_prob_values = []
    knndist_values = []
    knndist_l2_values = []
    network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(val_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size))

            features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
            means, log_sigma2s = network.head_net(features) # (both has shape: (batch_size, 1))
            means = means.view(-1) # (shape: (batch_size))
            log_sigma2s = log_sigma2s.view(-1) # (shape: (batch_size))
            sigma2s = torch.exp(log_sigma2s)
            sigmas = torch.sqrt(sigma2s)


            # (this is actually the log probability, not the probability)
            gmm_probs = feature_gmm.score_samples(features.cpu().numpy()) # (shape: (batch_size, ))
            gmm_prob_values += list(gmm_probs)


            neighbors_dist = []
            for i in range(features.size(0)):
                feature = features[i].cpu().numpy()
                _, train_dist = nns.ann_index.get_nns_by_vector(feature, 10, search_k=-1, include_distances=True)
                neighbors_dist.append(train_dist)
            neighbors_dist = np.array(neighbors_dist) # (shape: (batch_size, 10))
            knn_dists = np.mean(neighbors_dist, axis=1) # (shape: (batch_size, ))
            knndist_values += list(knn_dists)
            #
            neighbors_dist = []
            for i in range(features.size(0)):
                feature = features[i].cpu().numpy()
                _, train_dist = nns_l2.ann_index.get_nns_by_vector(feature, 10, search_k=-1, include_distances=True)
                neighbors_dist.append(train_dist)
            neighbors_dist = np.array(neighbors_dist) # (shape: (batch_size, 10))
            knn_dists = np.mean(neighbors_dist, axis=1) # (shape: (batch_size, ))
            knndist_l2_values += list(knn_dists)


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
    knndist_values = np.array(knndist_values)
    knndist_l2_values = np.array(knndist_l2_values)

    num_preds = abs_error_values.shape[0]
    #
    sorted_inds_gmm = np.argsort(list(-np.array(gmm_prob_values)))
    threshold_value_gmm_prob = gmm_prob_values[sorted_inds_gmm[int((0.95)*num_preds)]]
    #
    #
    sorted_inds_knndist = np.argsort(knndist_values)
    threshold_value_knndist = knndist_values[sorted_inds_knndist[int((0.95)*num_preds)]]
    #
    sorted_inds_knndist_l2 = np.argsort(knndist_l2_values)
    threshold_value_knndist_l2 = knndist_l2_values[sorted_inds_knndist_l2[int((0.95)*num_preds)]]

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
    knndist_values = []
    knndist_l2_values = []
    network.eval() # (set in eval mode, this affects BatchNorm and dropout)
    for step, (xs, ys) in enumerate(test_loader):
        with torch.no_grad():
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size))

            features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
            means, log_sigma2s = network.head_net(features) # (both has shape: (batch_size, 1))
            means = means.view(-1) # (shape: (batch_size))
            log_sigma2s = log_sigma2s.view(-1) # (shape: (batch_size))
            sigma2s = torch.exp(log_sigma2s)
            sigmas = torch.sqrt(sigma2s)


            # (this is actually the log probability, not the probability)
            gmm_probs = feature_gmm.score_samples(features.cpu().numpy()) # (shape: (batch_size, ))
            gmm_prob_values += list(gmm_probs)


            neighbors_dist = []
            for i in range(features.size(0)):
                feature = features[i].cpu().numpy()
                _, train_dist = nns.ann_index.get_nns_by_vector(feature, 10, search_k=-1, include_distances=True)
                neighbors_dist.append(train_dist)
            neighbors_dist = np.array(neighbors_dist) # (shape: (batch_size, 10))
            knn_dists = np.mean(neighbors_dist, axis=1) # (shape: (batch_size, ))
            knndist_values += list(knn_dists)
            #
            neighbors_dist = []
            for i in range(features.size(0)):
                feature = features[i].cpu().numpy()
                _, train_dist = nns_l2.ann_index.get_nns_by_vector(feature, 10, search_k=-1, include_distances=True)
                neighbors_dist.append(train_dist)
            neighbors_dist = np.array(neighbors_dist) # (shape: (batch_size, 10))
            knn_dists = np.mean(neighbors_dist, axis=1) # (shape: (batch_size, ))
            knndist_l2_values += list(knn_dists)


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
    knndist_values = np.array(knndist_values)
    knndist_l2_values = np.array(knndist_l2_values)

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


    knndist_sigma_values = sigma_values[knndist_values <= threshold_value_knndist]
    knndist_mean_values = mean_values[knndist_values <= threshold_value_knndist]
    knndist_y_values = y_values[knndist_values <= threshold_value_knndist]
    #
    knndist_pred_prop = float(knndist_y_values.shape[0])/float(y_values.shape[0])
    knndist_test_pred_props.append(knndist_pred_prop)
    #
    knndist_lower_values = knndist_mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*knndist_sigma_values
    knndist_upper_values = knndist_mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*knndist_sigma_values
    #
    knndist_lower_values = knndist_lower_values - Q_1_alpha ###########################################################################################
    knndist_upper_values = knndist_upper_values + Q_1_alpha ###########################################################################################
    #
    knndist_test_coverage = np.count_nonzero(np.logical_and(knndist_y_values >= knndist_lower_values, knndist_y_values <= knndist_upper_values))/float(knndist_y_values.shape[0])
    knndist_test_coverages.append(knndist_test_coverage)
    #
    knndist_test_avg_length = np.mean(knndist_upper_values - knndist_lower_values)
    knndist_test_avg_lengths.append(knndist_test_avg_length)


    knndist_l2_sigma_values = sigma_values[knndist_l2_values <= threshold_value_knndist_l2]
    knndist_l2_mean_values = mean_values[knndist_l2_values <= threshold_value_knndist_l2]
    knndist_l2_y_values = y_values[knndist_l2_values <= threshold_value_knndist_l2]
    #
    knndist_l2_pred_prop = float(knndist_l2_y_values.shape[0])/float(y_values.shape[0])
    knndist_l2_test_pred_props.append(knndist_l2_pred_prop)
    #
    knndist_l2_lower_values = knndist_l2_mean_values - scipy.stats.norm.ppf(1.0 - alpha/2)*knndist_l2_sigma_values
    knndist_l2_upper_values = knndist_l2_mean_values + scipy.stats.norm.ppf(1.0 - alpha/2)*knndist_l2_sigma_values
    #
    knndist_l2_lower_values = knndist_l2_lower_values - Q_1_alpha ###########################################################################################
    knndist_l2_upper_values = knndist_l2_upper_values + Q_1_alpha ###########################################################################################
    #
    knndist_l2_test_coverage = np.count_nonzero(np.logical_and(knndist_l2_y_values >= knndist_l2_lower_values, knndist_l2_y_values <= knndist_l2_upper_values))/float(knndist_l2_y_values.shape[0])
    knndist_l2_test_coverages.append(knndist_l2_test_coverage)
    #
    knndist_l2_test_avg_length = np.mean(knndist_l2_upper_values - knndist_l2_lower_values)
    knndist_l2_test_avg_lengths.append(knndist_l2_test_avg_length)


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
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    print ("knndist_test_pred_props:")
    print (knndist_test_pred_props)
    print ("knndist_test_pred_props: %g +/- %g" % (np.mean(np.array(knndist_test_pred_props)), np.std(np.array(knndist_test_pred_props))))
    print ("$")
    print ("$")
    print ("knndist_test_coverages:")
    print (knndist_test_coverages)
    print ("knndist_test_coverages: %g +/- %g" % (np.mean(np.array(knndist_test_coverages)), np.std(np.array(knndist_test_coverages))))
    print ("$")
    print ("$")
    print ("knndist_test_avg_lengths:")
    print (knndist_test_avg_lengths)
    print ("knndist_test_avg_lengths: %g +/- %g" % (np.mean(np.array(knndist_test_avg_lengths)), np.std(np.array(knndist_test_avg_lengths))))
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    print ("knndist_l2_test_pred_props:")
    print (knndist_l2_test_pred_props)
    print ("knndist_l2_test_pred_props: %g +/- %g" % (np.mean(np.array(knndist_l2_test_pred_props)), np.std(np.array(knndist_l2_test_pred_props))))
    print ("$")
    print ("$")
    print ("knndist_l2_test_coverages:")
    print (knndist_l2_test_coverages)
    print ("knndist_l2_test_coverages: %g +/- %g" % (np.mean(np.array(knndist_l2_test_coverages)), np.std(np.array(knndist_l2_test_coverages))))
    print ("$")
    print ("$")
    print ("knndist_l2_test_avg_lengths:")
    print (knndist_l2_test_avg_lengths)
    print ("knndist_l2_test_avg_lengths: %g +/- %g" % (np.mean(np.array(knndist_l2_test_avg_lengths)), np.std(np.array(knndist_l2_test_avg_lengths))))
    print ("###################################################################")
    print ("###################################################################")
    print ("###################################################################")
    print ("###################################################################")
