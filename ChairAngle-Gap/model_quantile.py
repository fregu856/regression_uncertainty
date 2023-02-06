# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.resnet import resnet34

import os

class QuantileHeadNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.fc1_low = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_low = nn.Linear(hidden_dim, 1)

        self.fc1_high = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_high = nn.Linear(hidden_dim, 1)

    def forward(self, x_feature):
        # (x_feature has shape: (batch_size, hidden_dim))

        low = F.relu(self.fc1_low(x_feature))  # (shape: (batch_size, hidden_dim))
        low = self.fc2_low(low)  # (shape: batch_size, 1))

        high = F.relu(self.fc1_high(x_feature))  # (shape: (batch_size, hidden_dim))
        high = self.fc2_high(high)  # (shape: batch_size, 1))

        return low, high


class QuantileFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet34 = resnet34(spectral_normalization=False)

    def forward(self, x):
        # (x has shape (batch_size, 3, img_size, img_size))

        x_feature = self.resnet34(x) # (shape: (batch_size, 512))

        return x_feature


class QuantileNet(nn.Module):
    def __init__(self, model_id, project_dir, alpha):
        super(QuantileNet, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        hidden_dim = 512

        self.feature_net = QuantileFeatureNet()
        self.head_net = QuantileHeadNet(hidden_dim)

    def forward(self, x, y):
        x_feature = self.feature_net(x) # (shape: (batch_size, hidden_dim))
        return self.head_net(x_feature)

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
