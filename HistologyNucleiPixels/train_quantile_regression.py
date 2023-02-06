# camera-ready

from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from model_quantile import QuantileNet

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

model_id = "quantile_regression"

alpha = 0.1
alpha_low = alpha/2.0
alpha_high = 1.0 - alpha/2.0
print(alpha)
print(alpha_low)
print(alpha_high)

num_epochs = 75
batch_size = 32
learning_rate = 0.001

def pinball_loss(y, y_hat, alpha_value):
    # (y and y_hat has shape: (batch_size))

    delta = y - y_hat # (shape: (batch_size))
    abs_delta = torch.abs(y - y_hat) # (shape: (batch_size))

    loss = torch.zeros_like(y) # (shape: (batch_size))
    loss[delta > 0] = alpha_value*abs_delta[delta > 0] # (shape: (batch_size))
    loss[delta <= 0] = (1.0 - alpha_value)*abs_delta[delta <= 0] # (shape: (batch_size))
    loss = torch.mean(loss)

    return loss


num_models = 20
for i in range(num_models):
    train_dataset = DatasetTrain()
    num_train_batches = int(len(train_dataset)/batch_size)
    print ("num_train_batches:", num_train_batches)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    network = QuantileNet(model_id + "_%d" % i, project_dir="/root/regression_uncertainty/HistologyNucleiPixels", alpha=alpha).cuda()

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

            lows, highs = network.head_net(x_features) # (both has shape: (batch_size, 1))
            lows = lows.view(-1) # (shape: (batch_size))
            highs = highs.view(-1) # (shape: (batch_size))

            ########################################################################
            # compute loss:
            ########################################################################
            if epoch < 20:
                loss = (torch.mean(torch.pow(ys - lows, 2)) + torch.mean(torch.pow(ys - highs, 2)))/2.0 # (burn-in)
            else:
                loss = (pinball_loss(ys, lows, alpha_low) + pinball_loss(ys, highs, alpha_high))/2.0

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

                print(torch.mean(lows).item())
                print(torch.mean(ys).item())
                print(torch.mean(highs).item())
                print(torch.mean(lows + (highs - lows)/2.0).item())
                print(torch.mean(highs - lows).item())

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
