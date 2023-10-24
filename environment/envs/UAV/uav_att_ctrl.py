from collector import data_collector
from FNTSMC import fntsmc_att, fntsmc_param
from uav import UAV, uav_param
from ref_cmd import *


class uav_att_ctrl(UAV):
    def __init__(self, UAV_param: uav_param, att_ctrl_param: fntsmc_param):
        super(uav_att_ctrl, self).__init__(UAV_param)
        self.att_ctrl = fntsmc_att(att_ctrl_param)
        self.collector = data_collector(round(self.time_max / self.dt))
        self.ref = np.zeros(3)
        self.dot_ref = np.zeros(3)

        '''参考轨迹记录'''
        self.ref_att_amplitude = None
        self.ref_att_period = None
        self.ref_att_bias_a = None
        self.ref_att_bias_phase = None
        self.att_trajectory = None
        '''参考轨迹记录'''

    def att_control(self, ref: np.ndarray, dot_ref: np.ndarray, dot2_ref):
        """
        @param ref:         参考信号
        @param dot_ref:     参考信号一阶导数
        @param dot2_ref:    参考信号二阶导数 (仅在姿态控制模式有效)
        @return:            Tx Ty Tz
        """
        self.ref = ref
        self.dot_ref = dot_ref
        e = self.rho1() - self.ref
        de = self.dot_rho1() - self.dot_ref
        sec_order_att_dy = self.second_order_att_dynamics()
        ctrl_mat = self.att_control_matrix()
        if dot2_ref is not None:
            self.att_ctrl.control_update(sec_order_att_dy, ctrl_mat, e, de, dot2_ref)
        else:
            self.att_ctrl.control_update(sec_order_att_dy, ctrl_mat, e, de, np.zeros(3))
        return self.att_ctrl.control

    def update(self, action: np.ndarray):
        """
        @param action:  三个力矩
        @return:
        """
        action_4_uav = np.insert(action, 0, self.m * self.g / (np.cos(self.phi) * np.cos(self.theta)))
        data_block = {'time': self.time,                    # simulation time
                      'control': action_4_uav,              # actual control command
                      'ref_angle': self.ref,                # reference angle
                      'ref_dot_angle': self.dot_ref,        # reference dot angle
                      'ref_pos': np.zeros(3),               # set to zero for attitude control
                      'ref_vel': np.zeros(3),               # set to zero for attitude control
                      'd_out': np.zeros(3),                 # set to zero for attitude control
                      'd_out_obs': np.zeros(3),             # set to zero for attitude control
                      'state': self.uav_state_call_back(),  # UAV state
                      'dot_angle': self.uav_dot_att()       # UAV dot angle
                      }
        self.collector.record(data_block)
        self.rk44(action=action_4_uav, dis=np.zeros(3), n=1, att_only=True)

    def generate_ref_att_trajectory(self, _amplitude: np.ndarray, _period: np.ndarray, _bias_a: np.ndarray, _bias_phase: np.ndarray):
        """
        @param _amplitude:
        @param _period:
        @param _bias_a:
        @param _bias_phase:
        @return:
        """
        t = np.linspace(0, self.time_max, int(self.time_max / self.dt) + 1)
        r_phi = _bias_a[0] + _amplitude[0] * np.sin(2 * np.pi / _period[0] * t + _bias_phase[0])
        r_theta = _bias_a[1] + _amplitude[1] * np.sin(2 * np.pi / _period[1] * t + _bias_phase[1])
        r_psi = _bias_a[2] + _amplitude[2] * np.sin(2 * np.pi / _period[2] * t + _bias_phase[2])
        return np.vstack((r_phi, r_theta, r_psi)).T

    def generate_random_att_trajectory(self, is_random: bool = False, yaw_fixed: bool = False):
        """
        @param is_random:	随机在振幅与周期
        @param yaw_fixed:	偏航角固定
        @return:			None
        """
        if is_random:
            A = np.array([
                np.random.uniform(low=0, high=self.phi_max - 10 * np.pi / 180),
                np.random.uniform(low=0, high=self.theta_max - 10 * np.pi / 180),
                np.random.uniform(low=0, high=self.psi_max - 10 * np.pi / 180)])
            T = np.random.uniform(low=3, high=6, size=3)  # 随机生成周期
            phi0 = np.random.uniform(low=0, high=np.pi / 2, size=3)
        else:
            A = np.array([np.pi / 3, np.pi / 3, np.pi / 2])
            T = np.array([5, 5, 5])
            phi0 = np.array([np.pi / 2, 0., 0.])

        if yaw_fixed:
            A[2] = 0.
            phi0[2] = 0.

        self.ref_att_amplitude = A
        self.ref_att_period = T
        self.ref_att_bias_a = np.zeros(3)
        self.ref_att_bias_phase = phi0
        self.att_trajectory = self.generate_ref_att_trajectory(self.ref_att_amplitude, self.ref_att_period, self.ref_att_bias_a, self.ref_att_bias_phase)

    def controller_reset(self):
        self.att_ctrl.fntsmc_att_reset()

    def controller_reset_with_new_param(self, new_att_param: fntsmc_param = None):
        if new_att_param is not None:
            self.att_ctrl.fntsmc_att_reset_with_new_param(new_att_param)

    def collector_reset(self, N: int):
        self.collector.reset(N)

    def reset_uav_att_ctrl(self,
                           random_att_trajectory: bool = False,
                           yaw_fixed: bool = False,
                           new_att_ctrl_param: fntsmc_param = None):
        """
        @param random_att_trajectory:
        @param yaw_fixed:
        @param new_att_ctrl_param:
        @return:
        """
        '''1. generate random trajectory'''
        self.generate_random_att_trajectory(is_random=random_att_trajectory, yaw_fixed=yaw_fixed)

        '''2. reset uav randomly or not'''
        self.reset_uav()

        '''3. reset collector'''
        self.collector_reset(round(self.time_max / self.dt))

        '''4. reset controller'''
        if new_att_ctrl_param is not None:
            self.att_ctrl.fntsmc_att_reset_with_new_param(new_att_ctrl_param)
        else:
            self.att_ctrl.fntsmc_att_reset()

        '''5. reset image'''
        self.draw_att_init_image()
