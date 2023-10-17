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
	data = pd.read_csv('./simulation/test_record.csv', header=0).to_numpy()
	# r = np.clip(data[:, 1], -500, 0)
	r1 = 0.2
	r2 = 1.0
	# print(data.shape)
	l = data.shape[0]
	print(l)
	i1 = int(l * r1)
	i2 = int(l * r2)
	plt.figure()
	plt.plot(data[i1: i2, 0], data[i1: i2, 1])
	plt.show()
	# a = np.array([1, 2, 3, 4, 5, -9, -10, np.pi])
	# print(np.min(a))
	# a = torch.ones((10, 2))
	# print(a)
	# # b = a.sum(dim=1, keepdim=True)
	# # print(b)
	# a[2] = 3
	# print(a)
