from uav import uav_param
from algorithm.rl_base.rl_base import rl_base
from uav_pos_ctrl import uav_pos_ctrl, fntsmc_param
import math
import numpy as np
from ref_cmd import *
from environment.Color import Color


class uav_pos_ctrl_RL(rl_base, uav_pos_ctrl):
	def __init__(self, _uav_param: uav_param, _uav_att_param: fntsmc_param, _uav_pos_param: fntsmc_param):
		rl_base.__init__(self)
		uav_pos_ctrl.__init__(self, _uav_param, _uav_att_param, _uav_pos_param)

		self.staticGain = 2.0

		'''state limitation'''
		# 并非要求数据一定在这个区间内，只是给一个归一化的系数而已，让 NN 不同维度的数据不要相差太大
		# 不要出现：某些维度的数据在 [-3, 3]，另外维度在 [0.05, 0.9] 这种情况即可
		self.e_pos_max = np.array([3., 3., 3.])
		self.e_pos_min = np.array([-3., -3., -3.])
		self.e_vel_max = np.array([3., 3., 3.])
		self.e_vel_min = np.array([-3., -3., -3.])
		'''state limitation'''

		'''rl_base'''
		self.name = 'uav_pos_ctrl_RL'
		self.state_dim = 3 + 3		# e_pos e_vel
		self.state_num = [math.inf for _ in range(self.state_dim)]
		self.state_step = [None for _ in range(self.state_dim)]
		self.state_space = [None for _ in range(self.state_dim)]
		self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
		self.isStateContinuous = [True for _ in range(self.state_dim)]

		self.initial_state = self.state_norm()
		self.current_state = self.initial_state.copy()
		self.next_state = self.initial_state.copy()

		self.action_dim = 3 + 3 + 1 + 1		# 3 for k1, 3 for k2, 1 for gamma, 1 for lambda
		self.action_step = [None for _ in range(self.action_dim)]
		self.action_range = [[0, 3.0] for _ in range(self.action_dim)]
		self.action_num = [math.inf for _ in range(self.action_dim)]
		self.action_space = [None for _ in range(self.action_dim)]
		self.isActionContinuous = [True for _ in range(self.action_dim)]
		self.initial_action = [0.0 for _ in range(self.action_dim)]
		self.current_action = self.initial_action.copy()

		self.reward = 0.
		self.Q_pos = np.array([1., 1., 1.])		# 位置误差惩罚
		self.Q_vel = np.array([0.01, 0.01, 0.01])	# 速度误差惩罚
		self.R = np.array([0.00, 0.00, 0.00])	# 期望加速度输出 (即控制输出) 惩罚
		self.is_terminal = False
		self.terminal_flag = 0
		'''rl_base'''

	def state_norm(self) -> np.ndarray:
		e_pos_norm = (self.uav_pos() - self.pos_ref) / (self.e_pos_max - self.e_pos_min) * self.staticGain
		e_vel_norm = (self.uav_vel() - self.dot_pos_ref) / (self.e_vel_max - self.e_vel_min) * self.staticGain

		norm_state = np.concatenate((e_pos_norm, e_vel_norm))
		return norm_state

	def inverse_state_norm(self) -> np.ndarray:
		inverse_e_pos_norm = self.current_state[0:3] / self.staticGain * (self.e_pos_max - self.e_pos_min)
		inverse_e_vel_norm = self.current_state[3:6] / self.staticGain * (self.e_vel_max - self.e_vel_min)

		inverse_norm_state = np.concatenate((inverse_e_pos_norm, inverse_e_vel_norm))
		return inverse_norm_state

	def get_reward(self, param=None):
		"""
		@param param:
		@return:
		"""
		ss = self.inverse_state_norm()
		_e_pos = ss[0: 3]
		_e_vel = ss[3: 6]

		'''reward for position error'''
		u_pos = -np.dot(_e_pos ** 2, self.Q_pos)

		'''reward for velocity error'''
		u_vel = -np.dot(_e_vel ** 2, self.Q_vel)

		'''reward for control output'''
		u_acc = -np.dot(self.pos_ctrl.control ** 2, self.R)

		'''reward for att out!!'''
		u_extra = 0.
		if self.terminal_flag == 3:
			print('角度出界了！！')
			_n = (self.time_max - self.time) / self.dt		# 剩余的步数
			_e = np.array([self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min])
			u0 = -np.dot(_e ** 2, self.Q_pos) * _n		# 给满位置惩罚
			u1 = -np.dot(_e_vel ** 2, self.Q_vel) * _n	# 给满速度惩罚
			u_extra = u0 + u1

		self.reward = u_pos + u_vel + u_acc + u_extra

	def is_success(self):
		"""
		@return:
		"""
		'''
			跟踪控制，暂时不定义 “成功” 的概念，不好说啥叫成功，啥叫失败
			因此置为 False，实际也不调用这个函数即可，学习不成功可考虑再加
		'''
		return False

	def is_Terminal(self, param=None):
		self.is_terminal, self.terminal_flag = self.is_episode_Terminal()

	def step_update(self, action: list):
		"""
		@param action:	这个 action 其实是油门 + 三个力矩
		@return:
		"""
		self.current_action = np.array(action)
		self.current_state = self.state_norm()

		self.update(action=self.current_action)
		self.is_Terminal()
		self.next_state = self.state_norm()
		self.get_reward()

	def get_param_from_actor(self, action_from_actor: np.ndarray):
		if np.min(action_from_actor) < 0:
			print('ERROR!!!!')
		for i in range(3):
			if action_from_actor[i] > 0:
				self.pos_ctrl.k1[i] = action_from_actor[i]
			if action_from_actor[i + 3] > 0:
				self.pos_ctrl.k2[i] = action_from_actor[i + 3]
		if action_from_actor[6] > 0:
			self.pos_ctrl.gamma[:] = action_from_actor[6]		# gamma gamma gamma
		if action_from_actor[7] > 0:
			self.pos_ctrl.lmd[:] = action_from_actor[7]		# lmd lmd lmd

	def reset_uav_pos_ctrl_RL_tracking(self,
									   random_trajectroy: bool = False,
									   random_pos0: bool = False,
									   new_att_ctrl_param: fntsmc_param = None,
									   new_pos_ctrl_parma: fntsmc_param = None):
		"""
		@param random_trajectroy:
		@param random_pos0:
		@param new_att_ctrl_param:
		@param new_pos_ctrl_parma:
		@return:
		"""
		self.reset_uav_pos_ctrl(random_trajectroy, random_pos0, new_att_ctrl_param, new_pos_ctrl_parma)

		'''RL_BASE'''
		self.initial_state = self.state_norm()
		self.current_state = self.initial_state.copy()
		self.next_state = self.initial_state.copy()
		self.current_action = self.initial_action.copy()
		self.reward = 0.
		self.is_terminal = False
		self.terminal_flag = 0
		'''RL_BASE'''
