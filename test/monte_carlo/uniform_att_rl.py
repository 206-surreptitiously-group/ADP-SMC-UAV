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
uav_param.pos_zone = np.atleast_2d([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-90), deg2rad(90)], [deg2rad(-90), deg2rad(90)], [deg2rad(-180), deg2rad(180)]])
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
	testPath0 = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/attitude_tracking/nets/draw_and_opt/'
	testPath_smc = None

	env = uav_att_ctrl_RL(uav_param, att_ctrl_param)
	env.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)

	env.load_norm_normalizer_from_file(testPath0, 'state_norm.csv')
	actor = PPOActor_Gaussian(state_dim=env.state_dim,
							  action_dim=env.action_dim,
							  a_min=np.array(env.action_range)[:, 0],
							  a_max=np.array(env.action_range)[:, 1],
							  init_std=0.5,
							  use_orthogonal_init=True)
	actor.orthogonal_init_all()
	actor.load_state_dict(torch.load(testPath0 + 'actor'))
	agent = PPO(env=env, actor=actor)

	r = []
	att_out = []

	CREATE_FILE = False
	# CONTROLLER = 'FNTSMC'
	CONTROLLER = 'RL'

	if CREATE_FILE:
		param = []
		A = np.linspace(0, deg2rad(80), 81)
		T = np.linspace(3,8,51)
		# phase_a = np.array([0., np.pi / 2, np.pi / 2])
		# bias_a = np.zeros(3)
		for _A in A:
			for _T in T:
				_param = [_A, _A, _A, _T, _T, _T, 0., np.pi / 2, np.pi / 2, 0., 0., 0.]
				param.append(_param)

		(pd.DataFrame(param,
					  columns=['A_phi', 'A_theta', 'A_psi', 'T_phi', 'T_theta', 'T_psi', 'phi0_phi', 'phi0_theta', 'phi0_psi', 'bias_phi', 'bias_theta', 'bias_psi']).
		 to_csv('./uniform_comparative/uniform_mc_ref_traj2.csv', sep=',', index=False))
	else:
		ref_trajectory = pd.read_csv('./uniform_comparative/A_0-80_T_3-8_phi0_0-90-0_bias_000/uniform_mc_ref_traj.csv', header=0).to_numpy()
		print(ref_trajectory.shape)
		cnt = 0
		for _traj in ref_trajectory:
			# 此文件仅仅测试 SMC 效果
			if cnt % 100 == 0:
				print('index: ', cnt)
			reset_att_ctrl_param('optimal')
			A = _traj[0: 3]
			T = _traj[3: 6]
			phi0 = _traj[6: 9]
			env.reset_uav_att_ctrl_RL_tracking(random_trajectroy=True,
											   yaw_fixed=False,
											   new_att_ctrl_param=att_ctrl_param,
											   outer_param=[A, T, phi0])
			test_r = 0.
			while not env.is_terminal:
				if CONTROLLER == 'RL':
					a = agent.evaluate(env.current_state_norm(env.current_state, update=False))
					# new_SMC_param = a.copy()
					env.get_param_from_actor(a)  # 将控制器参数更新
				rhod, dot_rhod, _, _ = ref_inner(env.time,
												   env.ref_att_amplitude,
												   env.ref_att_period,
												   env.ref_att_bias_a,
												   env.ref_att_bias_phase)
				torque = env.att_control(rhod, dot_rhod, None)
				env.step_update([torque[0], torque[1], torque[2]])
				test_r += env.reward

				# env.att_image = env.att_image_copy.copy()
				# env.draw_att(_rhod)
				# env.show_att_image(iswait=False)
			if env.terminal_flag == 3:
				print(A, T, phi0)
				print('==========')
			r.append(test_r)
			att_out.append(1 if env.terminal_flag == 3 else 0)
			cnt += 1
			if CONTROLLER == 'RL':
				(pd.DataFrame({'r': r, 'att_out': att_out}).
				 to_csv('./uniform_comparative/uniform_mc_test_rl.csv', sep=',', index=False))
			else:
				(pd.DataFrame({'r': r, 'att_out': att_out}).
				 to_csv('./uniform_comparative/uniform_mc_test_smc.csv', sep=',', index=False))
