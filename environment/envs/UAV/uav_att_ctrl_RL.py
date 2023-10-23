from uav import uav_param
from algorithm.rl_base.rl_base import rl_base
from uav_att_ctrl import uav_att_ctrl, fntsmc_param
import math
import numpy as np
from ref_cmd import *
from environment.Color import Color
from common.common_cls import Normalization
import pandas as pd


class uav_att_ctrl_RL(rl_base, uav_att_ctrl):
    def __init__(self, _uav_param: uav_param, _uav_att_param: fntsmc_param, _uav_pos_param: fntsmc_param):
        rl_base.__init__(self)
        uav_att_ctrl.__init__(self, _uav_param, _uav_att_param)

        self.staticGain = 2.0

        '''state limitation'''
        # 使用状态归一化，不用限制范围了
        '''state limitation'''

        '''rl_base'''
        self.name = 'uav_pos_ctrl_RL'
        self.state_dim = 3 + 3  # phi theta psi p q r
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]

        self.current_state_norm = Normalization(self.state_dim)
        self.next_state_norm = Normalization(self.state_dim)

        self.current_state = np.zeros(self.state_dim)
        self.next_state = np.zeros(self.state_dim)

        self.action_dim = 3 + 3 + 1 + 1  # 3 for k1, 3 for k2, 1 for gamma, 1 for lambda
        self.action_step = [None for _ in range(self.action_dim)]
        self.action_range = [[0, np.inf] for _ in range(self.action_dim)]
        self.action_num = [math.inf for _ in range(self.action_dim)]
        self.action_space = [None for _ in range(self.action_dim)]
        self.isActionContinuous = [True for _ in range(self.action_dim)]
        self.current_action = [0.0 for _ in range(self.action_dim)]

        self.reward = 0.
        self.Q_att = np.array([1., 1., 1.])  # 位置误差惩罚
        self.Q_pqr = np.array([0.05, 0.05, 0.05])  # 速度误差惩罚
        self.R = np.array([0.01, 0.01, 0.01])  # 期望加速度输出 (即控制输出) 惩罚
        self.is_terminal = False
        self.terminal_flag = 0
        '''rl_base'''

    def get_state(self) -> np.ndarray:
        e_att_ = self.uav_att() - self.ref
        e_pqr_ = self.uav_pqr() - self.dot_ref
        state = np.concatenate((e_att_, e_pqr_))
        return state

    def get_reward(self, param=None):
        """
		@param param:
		@return:
		"""
        _e_att = self.uav_att() - self.ref
        _e_pqr = self.uav_pqr() - self.dot_ref

        '''reward for position error'''
        u_att = -np.dot(_e_att ** 2, self.Q_att)

        '''reward for velocity error'''
        u_pqr = -np.dot(_e_pqr ** 2, self.Q_pqr)

        '''reward for control output'''
        u_acc = -np.dot(self.att_ctrl.control ** 2, self.R)

        '''reward for att out!!'''
        u_extra = 0.
        if self.terminal_flag == 2:  # 位置出界
            # print('Position out')
            '''
				给出界时刻的位置、速度、输出误差的累计
			'''
            _n = (self.time_max - self.time) / self.dt
            u_extra = _n * (u_att + u_pqr + u_acc)

        if self.terminal_flag == 3:
            print('Attitude out')
            '''
				给出界时刻的位置、速度、输出误差的累计
			'''
            _n = (self.time_max - self.time) / self.dt
            u_extra = _n * (u_att + u_pqr + u_acc)

        self.reward = u_att + u_pqr + u_acc + u_extra

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
        self.terminal_flag = self.get_terminal_flag()
        if self.terminal_flag == 0:  # 普通状态
            self.is_terminal = False
        elif self.terminal_flag == 1:  # 超时
            self.is_terminal = True
        elif self.terminal_flag == 2:  # 位置，姿态控制中不考虑位置变化
            self.is_terminal = False
        elif self.terminal_flag == 3:  # 姿态
            self.is_terminal = True
        else:
            self.is_terminal = False

    def step_update(self, action: list):
        """
		@param action:	这个 action 其实是油门 + 三个力矩
		@return:
		"""
        self.current_action = np.array(action)
        self.current_action[0] = self.m * self.g / (np.cos(self.phi) * np.cos(self.theta))      # 永远将油门设置为悬停
        self.current_state = self.get_state()

        self.update(action=self.current_action)
        self.is_Terminal()
        self.next_state = self.get_state()
        self.get_reward()

    def get_param_from_actor(self, action_from_actor: np.ndarray):
        """
        @param action_from_actor:
        @return:
        """
        if np.min(action_from_actor) < 0:
            print('ERROR!!!!')
        for i in range(3):
            if action_from_actor[i] > 0:
                self.att_ctrl.k1[i] = action_from_actor[i]      # k11 k12 k13
            if action_from_actor[i + 3] > 0:
                self.att_ctrl.k2[i] = action_from_actor[i + 3]  # k21 k22 k23
        if action_from_actor[6] > 0:
            self.att_ctrl.gamma[:] = action_from_actor[6]       # gamma gamma gamma
        if action_from_actor[7] > 0:
            self.att_ctrl.lmd[:] = action_from_actor[7]         # lmd lmd lmd

    def reset_uav_att_ctrl_RL_tracking(self,
                                       random_trajectroy: bool = False,
                                       yaw_fixed: bool = False,
                                       new_att_ctrl_param: fntsmc_param = None):
        """
        @param random_trajectroy:
        @param yaw_fixed:
        @param new_att_ctrl_param:
        @return:
        """
        self.reset_uav_att_ctrl(random_trajectroy, yaw_fixed, new_att_ctrl_param)

        '''RL_BASE'''
        self.current_state = self.get_state()
        self.next_state = self.get_state()
        self.current_action = self.initial_action.copy()
        self.reward = 0.
        self.is_terminal = False
        self.terminal_flag = 0
        '''RL_BASE'''

    def save_state_norm(self, path):
        data = {
            'cur_n': self.current_state_norm.running_ms.n * np.ones(self.state_dim),
            'cur_mean': self.current_state_norm.running_ms.mean,
            'cur_std': self.current_state_norm.running_ms.std,
            'cur_S': self.current_state_norm.running_ms.S,
            'next_n': self.next_state_norm.running_ms.n * np.ones(self.state_dim),
            'next_mean': self.next_state_norm.running_ms.mean,
            'next_std': self.next_state_norm.running_ms.std,
            'next_S': self.next_state_norm.running_ms.S,
        }
        pd.DataFrame(data).to_csv(path + 'state_norm.csv', index=False)

    def state_norm_batch(self, cur_data: np.ndarray, next_data: np.ndarray):
        ll = len(cur_data)
        for i in range(ll):
            cur_data[i] = self.current_state_norm(cur_data[i], update=True)
            next_data[i] = self.next_state_norm(next_data[i], update=True)
        return cur_data, next_data

    def load_norm_normalizer_from_file(self, path, file):
        data = pd.read_csv(path + file, header=0).to_numpy()
        self.current_state_norm.running_ms.n = data[0, 0]
        self.current_state_norm.running_ms.mean = data[:, 1]
        self.current_state_norm.running_ms.std = data[:, 2]
        self.current_state_norm.running_ms.S = data[:, 3]
        self.next_state_norm.running_ms.n = data[0, 4]
        self.next_state_norm.running_ms.mean = data[:, 5]
        self.next_state_norm.running_ms.std = data[:, 6]
        self.next_state_norm.running_ms.S = data[:, 7]
