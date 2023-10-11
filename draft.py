import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from environment.envs.UAV.uav import UAV, uav_param
from common.common_func import *
import cv2 as cv
import pandas


if __name__ == '__main__':
	data = pd.read_csv('test_record.csv', header=0).to_numpy()
	episode = data[:, 1]
	reward = data[:, 2]
	plt.figure()
	plt.plot(episode, reward)
	plt.xlabel('Episode')
	plt.ylabel('Reward per episode')
	plt.show()
