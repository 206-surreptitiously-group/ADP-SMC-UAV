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

    pass
