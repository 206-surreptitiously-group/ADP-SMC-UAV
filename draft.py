import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from environment.envs.UAV.uav import UAV, uav_param
from common.common_func import *
import cv2 as cv
import pandas
import torch.nn as nn


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
		super(PPOActorCritic, self).__init__()
		self.checkpoint_file = chkpt_dir + name + '_ppo'
		self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
		self.action_dim = _action_dim
		# 应该是初始化方差，一个动作就一个方差，两个动作就两个方差，std 是标准差
		self.action_var = torch.full((_action_dim,), _action_std_init * _action_std_init)
		self.actor = nn.Sequential(
			nn.Linear(_state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, _action_dim),
			nn.ReLU()  # 因为是参数优化，所以最后一层用ReLU
		)
		nn.init.orthogonal_(self.actor[0].weight, gain=1.0)
		nn.init.constant_(self.actor[0].bias, 0)
		nn.init.orthogonal_(self.actor[2].weight, gain=1.0)
		nn.init.constant_(self.actor[2].bias, 0)
		nn.init.orthogonal_(self.actor[4].weight, gain=1.0)
		nn.init.constant_(self.actor[4].bias, 0)
		self.critic = nn.Sequential(
			nn.Linear(_state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, 1)
		)
		nn.init.orthogonal_(self.critic[0].weight, gain=1.0)
		nn.init.constant_(self.critic[0].bias, 0)
		nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
		nn.init.constant_(self.critic[2].bias, 0)
		nn.init.orthogonal_(self.critic[4].weight, gain=1.0)
		nn.init.constant_(self.critic[4].bias, 0)




if __name__ == '__main__':
	data = pd.read_csv('./datasave/nets/test_record.csv', header=0).to_numpy()
	print(data.shape)
	plt.figure()
	plt.plot(data[:,0], data[:,1])
	plt.show()