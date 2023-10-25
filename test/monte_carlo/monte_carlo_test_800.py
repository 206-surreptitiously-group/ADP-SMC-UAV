import os
import sys
import datetime
import time
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.envs.UAV.uav_att_ctrl_RL import uav_att_ctrl_RL, uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from environment.envs.UAV.ref_cmd import *
from environment.Color import Color
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from algorithm.policy_base.Proximal_Policy_Optimization import PPOActor_Gaussian, PPOCritic
from common.common_cls import *
from common.common_func import *

timestep = 0
ENV = 'uav_pos_ctrl_RL'
ALGORITHM = 'PPO'

'''Parameter list of the quadrotor'''
DT = 0.02
uav_param = uav_param()
uav_param.m = 0.8
uav_param.g = 9.8
uav_param.J = np.array([4.212e-3, 4.212e-3, 8.255e-3])
uav_param.d = 0.12
uav_param.CT = 2.168e-6
uav_param.CM = 2.136e-8
uav_param.J0 = 1.01e-5
uav_param.kr = 1e-3
uav_param.kt = 1e-3
uav_param.pos0 = np.array([0, 0, 0])
uav_param.vel0 = np.array([0, 0, 0])
uav_param.angle0 = np.array([0, 0, 0])
uav_param.pqr0 = np.array([0, 0, 0])
uav_param.dt = DT
uav_param.time_max = 10
uav_param.pos_zone = np.atleast_2d([[-3, 3], [-3, 3], [0, 3]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-90), deg2rad(90)], [deg2rad(-90), deg2rad(90)], [deg2rad(-120), deg2rad(120)]])
'''Parameter list of the quadrotor'''

'''Parameter list of the attitude controller'''
att_ctrl_param = fntsmc_param()
att_ctrl_param.k1 = np.array([25, 25, 40])
att_ctrl_param.k2 = np.array([0.1, 0.1, 0.2])
att_ctrl_param.alpha = np.array([2.5, 2.5, 2.5])
att_ctrl_param.beta = np.array([0.99, 0.99, 0.99])
att_ctrl_param.gamma = np.array([1.5, 1.5, 1.2])
att_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
att_ctrl_param.dim = 3
att_ctrl_param.dt = DT
att_ctrl_param.ctrl0 = np.array([0., 0., 0.])
att_ctrl_param.saturation = np.array([0.3, 0.3, 0.3])
'''Parameter list of the attitude controller'''

test_episode = []
test_reward = []
sumr_list = []


def reset_att_ctrl_param(flag: str):
	if flag == 'zero':
		att_ctrl_param.k1 = 0.01 * np.ones(3)
		att_ctrl_param.k2 = 0.01 * np.ones(3)
		att_ctrl_param.gamma = 0.01 * np.ones(3)
		att_ctrl_param.lmd = 0.01 * np.ones(3)
	elif flag == 'random':
		att_ctrl_param.k1 = np.random.random(3)
		att_ctrl_param.k2 = np.random.random(3)
		att_ctrl_param.gamma = np.random.random() * np.ones(3)
		att_ctrl_param.lmd = np.random.random() * np.ones(3)
	else:  # optimal
		att_ctrl_param.k1 = np.array([25, 25, 40])
		att_ctrl_param.k2 = np.array([0.1, 0.1, 0.2])
		att_ctrl_param.gamma = np.array([1.5, 1.5, 1.2])
		att_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])


if __name__ == '__main__':
	# mc_path = os.path.dirname(os.path.abspath(__file__)) + '/monte_carlo/'
	# if not os.path.exists(mc_path):
	# 	os.makedirs(mc_path)
	mc_path = ''

	testPath0 = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/att_maybe_optimal1/'
	testPath_800 = testPath0 + 'trainNum_800/'

	env_test_800 = uav_att_ctrl_RL(uav_param, att_ctrl_param)
	env_test_800.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)
	env_test_800.load_norm_normalizer_from_file(testPath_800, 'state_norm.csv')

	actor_800 = PPOActor_Gaussian(state_dim=env_test_800.state_dim,
								  action_dim=env_test_800.action_dim,
								  a_min=np.array(env_test_800.action_range)[:, 0],
								  a_max=np.array(env_test_800.action_range)[:, 1],
								  init_std=0.5,
								  use_orthogonal_init=True)
	actor_800.orthogonal_init_all()
	actor_800.load_state_dict(torch.load(testPath_800 + 'actor_trainNum_' + str(800)))

	agent_800 = PPO(env=env_test_800, actor=actor_800)

	'''record for reward'''
	r_4_rl_800 = []
	'''record for reward'''

	'''record for attitude out'''
	att_out_rl_800 = []
	'''record for attitude out'''

	TEST_NUM = 2000
	CREATE_FILE = False

	if CREATE_FILE:
		A = np.stack((np.random.uniform(low=0, high=np.pi / 3, size=TEST_NUM),
					  np.random.uniform(low=0, high=np.pi / 3, size=TEST_NUM),
					  np.random.uniform(low=0, high=np.pi / 2, size=TEST_NUM))).T
		T = np.vstack((np.random.uniform(low=3, high=6, size=TEST_NUM),
					   np.random.uniform(low=3, high=6, size=TEST_NUM),
					   np.random.uniform(low=3, high=6, size=TEST_NUM))).T
		phase_a = np.vstack((np.random.uniform(low=0, high=np.pi / 2, size=TEST_NUM),
							 np.random.uniform(low=0, high=np.pi / 2, size=TEST_NUM),
							 np.random.uniform(low=0, high=np.pi / 2, size=TEST_NUM))).T
		bias_a = np.zeros((TEST_NUM, 3))

		(pd.DataFrame(np.hstack((A, T, phase_a, bias_a)),
					  columns=['A_phi', 'A_theta', 'A_psi', 'T_phi', 'T_theta', 'T_psi', 'phi0_phi', 'phi0_theta', 'phi0_psi', 'bias_phi', 'bias_theta', 'bias_psi']).
		 to_csv(mc_path + 'monte_carlo_ref_traj.csv', sep=',', index=False))
	else:
		ref_trajectory = pd.read_csv(mc_path + 'monte_carlo_ref_traj.csv', header=0).to_numpy()

		for i in range(TEST_NUM):
			if i % 100 == 0:
				print('i = ', i)
			# 此文件仅仅测试 training 800 效果
			reset_att_ctrl_param('zero')
			A = ref_trajectory[i][0: 3]
			T = ref_trajectory[i][3: 6]
			phi0 = ref_trajectory[i][6: 9]
			env_test_800.reset_uav_att_ctrl_RL_tracking(random_trajectroy=True,
														yaw_fixed=False,
														new_att_ctrl_param=att_ctrl_param,
														outer_param=[A, T, phi0])
			test_r = 0.
			att_out = 0
			while not env_test_800.is_terminal:
				a = agent_800.evaluate(env_test_800.current_state_norm(env_test_800.current_state, update=False))
				_new_SMC_param = a.copy()
				env_test_800.get_param_from_actor(_new_SMC_param)  # 将控制器参数更新
				_rhod, _dot_rhod, _, _ = ref_inner(env_test_800.time,
												   env_test_800.ref_att_amplitude,
												   env_test_800.ref_att_period,
												   env_test_800.ref_att_bias_a,
												   env_test_800.ref_att_bias_phase)
				_torque = env_test_800.att_control(_rhod, _dot_rhod, None)
				env_test_800.step_update([_torque[0], _torque[1], _torque[2]])
				test_r += env_test_800.reward

				# env_test_smc.att_image = env_test.att_image_copy.copy()
				# env_test_smc.draw_att(_rhod)
				# env_test_smc.show_att_image(iswait=False)
			r_4_rl_800.append(test_r)
			att_out_rl_800.append(1 if env_test_800.terminal_flag == 3 else 0)
		(pd.DataFrame({'r_4_rl_800': r_4_rl_800, 'att_out_rl_800': att_out_rl_800}).
		 to_csv(mc_path + 'monte_carlo_test_RL_800.csv', sep=',', index=False))
