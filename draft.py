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
	# data = pd.read_csv('./simulation/test_record.csv', header=0).to_numpy()
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
	optimal_SMC_params = np.atleast_2d(np.zeros(8))
	for i in range(10):
		optimal_SMC_params = np.insert(optimal_SMC_params, optimal_SMC_params.shape[0], (i + 1) * np.ones(8), axis=0)
	# print(optimal_SMC_params)
	(pd.DataFrame(data=optimal_SMC_params,
				 columns={'k11', 'k12', 'k13', 'k21', 'k22', 'k23', 'gamma', 'lambda'})
	 .to_csv('yyf.csv', sep=',', index=False))
