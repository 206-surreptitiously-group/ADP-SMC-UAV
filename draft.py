import numpy as np
import matplotlib.pyplot as plt
import torch
from environment.envs.UAV.uav import UAV, uav_param
from common.common_func import *
import cv2 as cv


if __name__ == '__main__':
	# a = np.array([1, 2, 3, 4, 5, 6, 7, 8]).astype(float)
	# k1 = np.zeros(3)
	# k2 = np.zeros(3)
	# gamma = np.zeros(3)
	# lmd = np.zeros(3)
	#
	# k1[:] = a[0:3]
	# k2[:] = a[3:6]
	# gamma = a[6] * np.ones(3)
	# lmd = a[7] * np.ones(3)
	#
	# print(k1)
	# print(k2)
	# print(gamma)
	# print(lmd)
	a = torch.randn(5)
	print(a)
	a = torch.clip(a, 0, torch.inf)
	print(a)
