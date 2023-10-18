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

	# data = {
	# 	'cur_n': 1*np.ones(5),
	# 	'cur_mean': 2*np.ones(5),
	# 	'cur_std': 3*np.ones(5),
	# 	'cur_S': 4*np.ones(5),
	# 	'next_n': 5*np.ones(5),
	# 	'next_mean': 6*np.ones(5),
	# 	'next_std': 7*np.ones(5),
	# 	'next_S': 8*np.ones(5),
	# }
	# pd.DataFrame(data).to_csv('state_norm.csv', index=False)

	# a = 10 * torch.randn(6)
	# a1 = 3 * torch.ones(6)
	# a2 = 8 * torch.ones(6)
	# b = torch.maximum(torch.minimum(a, a2), a1)
	# print(a)
	# print(b)
	log_std = nn.Parameter(np.log(0.5) * torch.ones(8))
