import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from environment.envs.UAV.uav import UAV, uav_param
from common.common_func import *
import cv2 as cv
import pandas


if __name__ == '__main__':
	data = {'episode': np.array([10, 20, 30, 40]), 'reward': np.array([10, 20, 30, 40])}
	pd.DataFrame(data).to_csv('record.csv')
