# camera-ready

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.resnet import resnet34

import os

class GaussHeadNet(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.fc1_mean = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, 1)

        self.fc1_var = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_var = nn.Linear(hidden_dim, 1)

    def forward(self, x_feature):
        # (x_feature has shape: (batch_size, hidden_dim))

        mean = F.relu(self.fc1_mean(x_feature))  # (shape: (batch_size, hidden_dim))
        mean = self.fc2_mean(mean)  # (shape: batch_size, 1))

        log_var = F.relu(self.fc1_var(x_feature))  # (shape: (batch_size, hidden_dim))
        log_var = self.fc2_var(log_var)  # (shape: batch_size, 1))

        return mean, log_var


class GaussFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet34 = resnet34(spectral_normalization=False)

    def forward(self, x):
        # (x has shape (batch_size, 3, img_size, img_size))

        x_feature = self.resnet34(x) # (shape: (batch_size, 512))

        return x_feature


class GaussNet(nn.Module):
    def __init__(self, model_id, project_dir):
        super(GaussNet, self).__init__()

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        hidden_dim = 512

        self.feature_net = GaussFeatureNet()
        self.head_net = GaussHeadNet(hidden_dim)

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
