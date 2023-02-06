# camera-ready

from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model_gaussian import GaussNet

import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torch.distributions

import math
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

model_id = "gaussian"

num_epochs = 75
batch_size = 32
learning_rate = 0.001

num_models = 20
for i in range(num_models):
    train_dataset = DatasetTrain()
    num_train_batches = int(len(train_dataset)/batch_size)
    print ("num_train_batches:", num_train_batches)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    network = GaussNet(model_id + "_%d" % i, project_dir="/root/regression_uncertainty/ChairAngle-Tails").cuda()

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    epoch_losses_train = []
    for epoch in range(num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("model: %d/%d  |  epoch: %d/%d" % (i+1, num_models, epoch+1, num_epochs))


        ########################################################################
        # train:
        ########################################################################
        network.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses_train = []
        for step, (xs, ys) in enumerate(train_loader):
            xs = xs.cuda() # (shape: (batch_size, 3, img_size, img_size))
            ys = ys.cuda() # (shape: (batch_size))

            x_features = network.feature_net(xs) # (shape: (batch_size, hidden_dim))
            if (epoch >= 20) and (epoch < 25):
                ####################################################################
                # make sure we do NOT train the resnet feature extractor:
                ####################################################################
                x_features = x_features.detach()
                ####################################################################

            means, log_sigma2s = network.head_net(x_features) # (both has shape: (batch_size, 1))
            means = means.view(-1) # (shape: (batch_size))
            log_sigma2s = log_sigma2s.view(-1) # (shape: (batch_size))

            ########################################################################
            # compute loss:
            ########################################################################
            if epoch < 20:
                loss = torch.mean(torch.pow(ys - means, 2))
            else:
                loss = torch.mean(torch.exp(-log_sigma2s)*torch.pow(ys - means, 2) + log_sigma2s)

            loss_value = loss.data.cpu().numpy()
            batch_losses_train.append(loss_value)

            ########################################################################
            # optimization step:
            ########################################################################
            optimizer.zero_grad() # (reset gradients)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)

            if step % 10 == 0:
                print ("model: %d/%d  |  epoch: %d/%d  |  step: %d/%d  |  loss: %g" % (i+1, num_models, epoch+1, num_epochs, step+1, num_train_batches, loss_value))

        epoch_loss_train = np.mean(batch_losses_train)
        epoch_losses_train.append(epoch_loss_train)
        with open("%s/epoch_losses_train.pkl" % network.model_dir, "wb") as file:
            pickle.dump(epoch_losses_train, file)
        print ("train loss: %g" % epoch_loss_train)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % network.model_dir)
        plt.close(1)
        #
        if epoch > 20:
            plt.figure(1)
            plt.plot(epoch_losses_train, "k^")
            plt.plot(epoch_losses_train, "k")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.ylim([np.min(np.array(epoch_losses_train[21:])), np.max(np.array(epoch_losses_train[21:]))])
            plt.title("train loss per epoch")
            plt.savefig("%s/epoch_losses_after_init_train.png" % network.model_dir)
            plt.close(1)

    # save the model weights to disk:
    checkpoint_path = network.checkpoints_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
    torch.save(network.state_dict(), checkpoint_path)
