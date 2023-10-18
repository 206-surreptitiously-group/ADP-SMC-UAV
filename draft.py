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
from torch.distributions import Normal


if __name__ == '__main__':
	# data = pd.read_csv('./simulation/test_record_TD3.csv', header=0).to_numpy()
	# r = data[:, 1]
	# print(r.shape)
	# print(np.max(r))
	# print(np.argmax(r))
	# r = np.clip(data[:, 1], -500, 0)
	# r1 = 0.2
	# r2 = 1.0
	# # print(data.shape)
	# l = data.shape[0]
	# print(l)
	# i1 = int(l * r1)
	# i2 = int(l * r2)
	# plt.figure()
	# plt.plot(data[i1: i2, 0], data[i1: i2, 1])
	# plt.show()

	std = torch.tensor([0.2, 1, 0], dtype=torch.float)
	mean = torch.ones(3)
	# print(mean)
	# print(std)
	# std.expand_as(mean)
	# print(std)
	print(mean.size())
	print(std.size())
	dist = Normal(mean, std)
	print(dist.sample())
