import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from environment.envs.UAV.uav import UAV, uav_param
from common.common_func import *
import cv2 as cv
import pandas
import torch.nn as nn
import argparse
from torch.distributions import Normal, MultivariateNormal


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PPOActor_Gaussian(nn.Module):
    def __init__(self,
                 a_min: np.ndarray = np.zeros(8),
                 a_max: np.ndarray = np.ones(8),
                 init_std: float = 0.5,
                 use_orthogonal_init: bool = True):
        super(PPOActor_Gaussian, self).__init__()
        '''k11, k21'''
        self.x1 = nn.Linear(2, 8)
        self.x2 = nn.Linear(8, 8)
        self.x3 = nn.Linear(8, 2)

        '''k12, k22'''
        self.y1 = nn.Linear(2, 8)
        self.y2 = nn.Linear(8, 8)
        self.y3 = nn.Linear(8, 2)

        '''k13, k23'''
        self.z1 = nn.Linear(2, 8)
        self.z2 = nn.Linear(8, 8)
        self.z3 = nn.Linear(8, 2)

        self.all1 = nn.Linear(6, 16)
        self.all2 = nn.Linear(16, 16)
        self.all3 = nn.Linear(16, 2)

        # self.mean_layer = nn.Linear(32, action_dim)
        self.activate_func = nn.Tanh()
        self.a_min = torch.tensor(a_min, dtype=torch.float)
        self.a_max = torch.tensor(a_max, dtype=torch.float)
        self.off = (self.a_min + self.a_max) / 2.0
        self.gain = self.a_max - self.off
        self.action_dim = 8
        self.std = torch.tensor(init_std, dtype=torch.float)

        if use_orthogonal_init:
            # print("------use_orthogonal_init------")
            orthogonal_init(self.x1)
            orthogonal_init(self.x2)
            orthogonal_init(self.x3, gain=0.01)
            orthogonal_init(self.y1)
            orthogonal_init(self.y2)
            orthogonal_init(self.y3, gain=0.01)
            orthogonal_init(self.z1)
            orthogonal_init(self.z2)
            orthogonal_init(self.z3, gain=0.01)
            orthogonal_init(self.all1)
            orthogonal_init(self.all2)
            orthogonal_init(self.all3, gain=0.01)
            # orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        o1 = s[:, [0, 3]]
        o1 = self.activate_func(self.x1(o1))
        o1 = self.activate_func(self.x2(o1))
        o1 = self.x3(o1)

        o2 = s[:, [1, 4]]
        o2 = self.activate_func(self.y1(o2))
        o2 = self.activate_func(self.y2(o2))
        o2 = self.y3(o2)

        o3 = s[:, [2, 5]]
        o3 = self.activate_func(self.z1(o3))
        o3 = self.activate_func(self.z2(o3))
        o3 = self.z3(o3)

        o4 = self.activate_func(self.all1(s))
        o4 = self.activate_func(self.all2(o4))
        o4 = self.all3(o4)

        o = torch.cat((o1, o2, o3, o4), 1)
        o = torch.tanh(o)  * self.gain + self.off
        return o

    def get_dist(self, s):
        mean = self.forward(s)
        # mean = torch.tensor(mean, dtype=torch.float)
        # log_std = self.log_std.expand_as(mean)
        # std = torch.exp(log_std)
        std = self.std.expand_as(mean)
        dist = Normal(mean, std)  # Get the Gaussian distribution
        # std = self.std.expand_as(mean)
        # dist = Normal(mean, std)
        return dist


if __name__ == '__main__':
    # path1 = './datasave/nets/train_opt2/test_record.csv'
    # data = pd.read_csv('./sumr_list.csv', header=0).to_numpy()
    #
    # # data[:, 1] = np.clip(data[:, 1], -50000, 0)
    # r1 = 0.0
    # r2 = 1.0
    # # print(data.shape)
    # l = data.shape[0]
    # print(l)
    # i1 = int(l * r1)
    # i2 = int(l * r2)
    # plt.figure()
    # plt.plot(data[i1: i2, 0], data[i1: i2, 1])
    # plt.show()

    # actor = PPOActor_Gaussian()
    # s = torch.rand((100, 6))
    # actor.forward(s)
    print(np.arange(10))
    pass
