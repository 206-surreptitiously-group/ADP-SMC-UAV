import numpy as np
from uav import UAV, uav_param
from collector import data_collector
from FNTSMC import fntsmc_att, fntsmc_pos, fntsmc_param
from ref_cmd import *
from environment.Color import Color


class uav_pos_ctrl(UAV):
	def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param, pos_ctrl_param: fntsmc_param):
		super(uav_pos_ctrl, self).__init__(UAV_param)
		self.att_ctrl = fntsmc_att(att_ctrl_param)
		self.pos_ctrl = fntsmc_pos(pos_ctrl_param)

		self.collector = data_collector(round(self.time_max / self.dt))
		self.collector.reset(round(self.time_max / self.dt))

		self.pos_ref = np.zeros(3)
		self.dot_pos_ref = np.zeros(3)
		self.att_ref = np.zeros(3)
		self.att_ref_old = np.zeros(3)
		self.dot_att_ref = np.zeros(3)

		self.obs = np.zeros(3)			# output of the observer
		self.dis = np.zeros(3)			# external disturbance, known by me, but not the controller

		'''参考轨迹记录'''
		self.ref_amplitude = None
		self.ref_period = None
		self.ref_bias_a = None
		self.ref_bias_phase = None
		'''参考轨迹记录'''

	def pos_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray, dis: np.ndarray, obs: np.ndarray):
		"""
        @param dis:         external disturbance
        @param ref:         x_d y_d z_d
        @param dot_ref:     vx_d vy_d vz_d
        @param dot2_ref:    ax_d ay_d az_d
        @param obs:         observer
        @return:            ref_phi ref_theta throttle
        """
		self.pos_ref = ref
		self.dot_pos_ref = dot_ref
		self.dis = dis
		self.obs = obs
		e = self.eta() - ref
		de = self.dot_eta() - dot_ref
		self.pos_ctrl.control_update(self.kt, self.m, self.uav_vel(), e, de, dot2_ref, obs)
		phi_d, theta_d, uf = self.uo_2_ref_angle_throttle(limit=[np.pi / 3, np.pi / 3], att_limitation=True)
		return phi_d, theta_d, uf

	def att_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref: np.ndarray, att_only: bool = False):
		"""
        @param ref:         参考信号
        @param dot_ref:     参考信号一阶导数
        @param dot2_ref:    参考信号二阶导数 (仅在姿态控制模式有效)
        @param att_only:    为 True 时，dot2_ref 正常输入
                            为 True 时，dot2_ref 为 0
        @return:            Tx Ty Tz
        """
		self.att_ref_old = self.att_ref.copy()
		self.att_ref = ref
		self.dot_att_ref = dot_ref
		if not att_only:
			dot2_ref = np.zeros(3)

		e = self.rho1() - ref
		de = self.dot_rho1() - dot_ref
		sec_order_att_dy = self.second_order_att_dynamics()
		ctrl_mat = self.att_control_matrix()
		self.att_ctrl.control_update(sec_order_att_dy, ctrl_mat, e, de, dot2_ref)
		return self.att_ctrl.control

	def uo_2_ref_angle_throttle(self, limit = None, att_limitation: bool = False):
		ux = self.pos_ctrl.control[0]
		uy = self.pos_ctrl.control[1]
		uz = self.pos_ctrl.control[2]
		uf = (uz + self.g) * self.m / (np.cos(self.phi) * np.cos(self.theta))
		asin_phi_d = min(max((ux * np.sin(self.psi) - uy * np.cos(self.psi)) * self.m / uf, -1), 1)
		phi_d = np.arcsin(asin_phi_d)
		asin_theta_d = min(max((ux * np.cos(self.psi) + uy * np.sin(self.psi)) * self.m / (uf * np.cos(phi_d)), -1), 1)
		theta_d = np.arcsin(asin_theta_d)
		if att_limitation:
			if limit is not None:
				phi_d = max(min(phi_d, limit[0]), -limit[0])
				theta_d = max(min(theta_d, limit[1]), -limit[1])
		return phi_d, theta_d, uf

	def update(self, action: np.ndarray):
		"""
        @param action:  油门 + 三个力矩
        @return:
        """
		data_block = {'time': self.time,  # simulation time
					  'control': action,  # actual control command
					  'ref_angle': self.att_ref,  # reference angle
					  'ref_pos': self.pos_ref,
					  'ref_vel': self.dot_pos_ref,
					  'd_out': self.dis / self.m,
					  'd_out_obs': self.obs,
					  'state': self.uav_state_call_back()}  # quadrotor state
		self.collector.record(data_block)
		self.rk44(action=action, dis=self.dis, n=1, att_only=False)

	def generate_ref_trajectory(self, _amplitude: np.ndarray, _period: np.ndarray, _bias_a: np.ndarray, _bias_phase: np.ndarray):
		"""
        @param _amplitude:
        @param _period:
        @param _bias_a:
        @param _bias_phase:
        @return:
        """
		t = np.linspace(0, self.time_max, int(self.time_max / self.dt) + 1)
		rx = _bias_a[0] + _amplitude[0] * np.sin(2 * np.pi / _period[0] * t + _bias_phase[0])
		ry = _bias_a[1] + _amplitude[1] * np.sin(2 * np.pi / _period[1] * t + _bias_phase[1])
		rz = _bias_a[2] + _amplitude[2] * np.sin(2 * np.pi / _period[2] * t + _bias_phase[2])
		rpsi = _bias_a[3] + _amplitude[3] * np.sin(2 * np.pi / _period[3] * t + _bias_phase[3])
		return np.vstack((rx, ry, rz, rpsi)).T

	def generate_random_set_point(self, is_random: bool = False, pos: np.ndarray = np.zeros, yaw_fixed: bool = True):
		"""
		@param is_random:
		@param pos:
		@return:
		"""
		if is_random:
			_ref_bias_a = np.random.uniform(low=np.concatenate((self.pos_zone[:, 0], [self.att_zone[2][0]])),
											high=np.concatenate((self.pos_zone[:, 1], [self.att_zone[2][1]])),
											size=4)
		else:
			_yaw = np.random.uniform(low=self.att_zone[2][0], high=self.att_zone[2][1])
			_ref_bias_a = np.concatenate((pos, [_yaw]))

		if yaw_fixed:
			_ref_bias_a[3] = 0.
		self.ref_amplitude = np.zeros(4)
		self.ref_period = np.ones(4)
		self.ref_bias_a = _ref_bias_a
		self.ref_bias_phase = np.zeros(4)

	def generate_random_trajectory(self, is_random: bool = False, yaw_fixed: bool = True):
		"""
		@param is_random:	随机在振幅与周期
		@param yaw_fixed:	偏航角固定
		@return:			None
		"""
		if is_random:
			rxy = np.random.uniform(low=0, high=3, size=2)  # 随机生成 xy 方向振幅
			rz = np.random.uniform(low=0, high=1.5)  # 随机生成 z 方向振幅
			rpsi = np.random.uniform(low=0, high=np.pi / 2)

			T = np.random.uniform(low=5, high=10, size=4)  # 随机生成周期
			phase_xyzpsi = np.random.uniform(low=0, high=np.pi / 2, size=4)
		else:
			rxy = np.array([1.5, 1.5])
			rz = 0.3
			rpsi = 0.

			T = np.array([6., 6., 10, 10])

			phase_xyzpsi = np.array([np.pi / 2, 0., 0., 0.])

		if yaw_fixed:
			rpsi = 0.
			phase_xyzpsi[3] = 0.

		self.ref_amplitude = np.array([rxy[0], rxy[1], rz, rpsi])  # x y z psi
		self.ref_period = T
		self.ref_bias_a = np.array([0, 0, 1.5, 0])
		self.ref_bias_phase = phase_xyzpsi
		self.trajectory = self.generate_ref_trajectory(self.ref_amplitude, self.ref_period, self.ref_bias_a, self.ref_bias_phase)

	def generate_random_start_target(self):
		x = np.random.uniform(low=self.pos_zone[0][0], high=self.pos_zone[0][1], size=2)
		y = np.random.uniform(low=self.pos_zone[1][0], high=self.pos_zone[1][1], size=2)
		z = np.random.uniform(low=self.pos_zone[2][0], high=self.pos_zone[2][1], size=2)
		psi = np.random.uniform(low=self.att_zone[2][0], high=self.att_zone[2][1], size=2)
		st = np.vstack((x, y, z, psi))
		start = st[:, 0]
		target = st[:, 1]
		return start, target

	def controller_reset(self):
		self.att_ctrl.fntsmc_att_reset()
		self.pos_ctrl.fntsmc_pos_reset()

	def controller_reset_with_new_param(self, new_att_param: fntsmc_param = None, new_pos_param: fntsmc_param = None):
		if new_att_param is not None:
			self.att_ctrl.fntsmc_att_reset_with_new_param(new_att_param)
		if new_pos_param is not None:
			self.pos_ctrl.fntsmc_pos_reset_with_new_param(new_pos_param)

	def collector_reset(self, N: int):
		self.collector.reset(N)

	def set_random_init_pos(self, pos0: np.ndarray, r: np.ndarray):
		"""
        @brief:         为无人机设置随机的初始位置
        @param pos0:    参考轨迹第一个点
        @param r:       容许半径
        @return:        无人机初始点
        """
		return np.random.uniform(low=pos0 - np.fabs(r), high=pos0 + np.fabs(r), size=3)

	def generate_action_4_uav(self):
		ref, dot_ref, dot2_ref, _ = ref_uav(self.time, self.ref_amplitude, self.ref_period, self.ref_bias_a, self.ref_bias_phase)
		uncertainty = generate_uncertainty(time=self.time, is_ideal=True)
		obs = np.zeros(3)

		phi_d, theta_d, throttle = self.pos_control(ref[0:3], dot_ref[0:3], dot2_ref[0:3], uncertainty, obs)
		dot_phi_d = (phi_d - self.att_ref[0]) / self.dt
		dot_theta_d = (theta_d - self.att_ref[1]) / self.dt

		rho_d = np.array([phi_d, theta_d, ref[3]])
		dot_rho_d = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])
		torque = self.att_control(rho_d, dot_rho_d, np.zeros(3), att_only=False)
		action_4_uav = [throttle, torque[0], torque[1], torque[2]]

		return action_4_uav

	def reset_uav_pos_ctrl(self,
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
		'''1. generate random trajectory'''
		self.generate_random_trajectory(is_random=random_trajectroy, yaw_fixed=True)

		'''2. reset uav randomly or not'''
		if random_pos0:
			_param = self.get_param_from_uav()
			_param.pos0 = self.set_random_init_pos(pos0=self.trajectory[0][0:3], r=0.3 * np.ones(3))
			self.reset_uav_with_param(_param)
		else:
			self.reset_uav()

		'''3. reset collector'''
		self.collector_reset(round(self.time_max / self.dt))

		'''4. reset controller'''
		if new_att_ctrl_param is not None:
			self.att_ctrl.fntsmc_att_reset_with_new_param(new_att_ctrl_param)
		else:
			self.att_ctrl.fntsmc_att_reset()

		if new_pos_ctrl_parma is not None:
			self.pos_ctrl.fntsmc_pos_reset_with_new_param(new_pos_ctrl_parma)
		else:
			self.pos_ctrl.fntsmc_pos_reset()

		'''5. reset iamge'''
		self.draw_3d_trajectory_projection(self.trajectory)
		self.draw_init_image()

	def RISE(self, offset: float = 0.1):
		offset = max(min(offset, 0.4), 0.1)
		index = self.collector.index
		i1 = int(offset * index)
		i2 = int((1 - offset) * index)
		ref = self.collector.ref_pos[i1: i2, :]  # n * 3
		pos = self.collector.state[i1: i2, 0: 3]  # n * 3

		rise_x = np.sqrt(np.sum((ref[:, 0] - pos[:, 0]) ** 2) / len(ref[:, 0]))
		rise_y = np.sqrt(np.sum((ref[:, 1] - pos[:, 1]) ** 2) / len(ref[:, 1]))
		rise_z = np.sqrt(np.sum((ref[:, 2] - pos[:, 2]) ** 2) / len(ref[:, 2]))

		return np.array([rise_x, rise_y, rise_z])
