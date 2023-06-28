# import numpy as np
# from environment.Color import Color
import numpy as np

from environment.utils import *


class UAV:
    def __init__(self,
                 m: float = 0.8,
                 g: float = 9.8,
                 J: np.ndarray = np.array([4.212e-3, 4.212e-3, 8.255e-3]),
                 d: float = 0.12,
                 CT: float = 3.138e-6,
                 CM: float = 2.136e-8,
                 J0: float = 1.01e-5,
                 kr: float = 6e-3,
                 kt: float = 6e-3,
                 pos0=None,
                 vel0=None,
                 angle0=None,
                 omega0_body=None
                 ):
        if omega0_body is None:
            omega0_body = np.array([0, 0, 0])
        if angle0 is None:
            angle0 = np.array([0, 0, 0])
        if vel0 is None:
            vel0 = np.array([0, 0, 0])
        if pos0 is None:
            pos0 = np.array([0, 0, 0])

        '''physical parameters'''
        self.m = m  # 无人机质量
        self.g = g  # 重力加速度
        self.J = J  # 转动惯量
        self.d = d  # 机臂长度 'X'构型
        self.CT = CT  # 螺旋桨升力系数
        self.CM = CM  # 螺旋桨力矩系数
        self.J0 = J0  # 电机和螺旋桨的转动惯量
        self.kr = kr  # 旋转阻尼系数
        self.kt = kt  # 平移阻尼系数

        self.x = pos0[0]
        self.y = pos0[1]
        self.z = pos0[2]

        self.vx = vel0[0]
        self.vy = vel0[1]
        self.vz = vel0[2]

        self.phi = angle0[0]
        self.theta = angle0[1]
        self.psi = angle0[2]

        self.p = omega0_body[0]
        self.q = omega0_body[1]
        self.r = omega0_body[2]

        # '''top left-1 top right-2 bottom-right-3 bottom-left-4'''
        # self.power_allocation_mat = \
        #     np.array([[CT, CT, CT, CT],
        #               [CT * d / np.sqrt(2), -CT * d / np.sqrt(2), -CT * d / np.sqrt(2), CT * d / np.sqrt(2)],
        #               [-CT * d / np.sqrt(2), -CT * d / np.sqrt(2), CT * d / np.sqrt(2), CT * d / np.sqrt(2)],
        #               [-CM, CM, -CM, CM]])  # 这个矩阵满秩

        '''top right-1 bottom left-2 top left-3 bottom-right-4'''
        self.power_allocation_mat = \
            np.array([[CT, CT, CT, CT],
                      [-CT * d / np.sqrt(2), CT * d / np.sqrt(2), CT * d / np.sqrt(2), -CT * d / np.sqrt(2)],
                      [CT * d / np.sqrt(2), -CT * d / np.sqrt(2), CT * d / np.sqrt(2), -CT * d / np.sqrt(2)],
                      [-CM, -CM, CM, CM]])  # 这个矩阵满秩

        self.dt = 0.001
        self.n = 0  # 记录走过的拍数
        self.time = 0.  # 当前时间
        self.time_max = 25  # 每回合最大时间
        '''physical parameters'''

        '''uncertainty and external disturbances'''
        # self.uncertainty = np.array([0, 0, 0, 0, 0, 0]).astype(float)
        # Fdx, Fdy, Fyz, dp, dq, dr
        '''uncertainty and external disturbances'''

        '''control'''
        self.throttle = self.m * self.g  # 油门
        self.torque = np.array([0., 0., 0.]).astype(float)  # 转矩

        self.control_ideal = np.array([self.m * self.g, 0., 0., 0.]).astype(float)  # 控制量
        self.control = self.control_ideal.copy()

        self.w_rotor_ideal = np.sqrt(np.dot(np.linalg.inv(self.power_allocation_mat), self.control_ideal))  # 理想电机转速
        self.w_rotor = self.w_rotor_ideal.copy()
        '''control'''

        'state limitation'
        self.pos_min = np.array([-10, -10, -10])
        self.pos_max = np.array([10, 10, 10])
        self.vel_min = np.array([-10, -10, -10])
        self.vel_max = np.array([10, 10, 10])
        self.angle_min = np.array([-deg2rad(80), -deg2rad(80), -deg2rad(180)])
        self.angle_max = np.array([deg2rad(80), deg2rad(80), deg2rad(180)])
        self.dangle_min = np.array([-deg2rad(360 * 3), -deg2rad(360 * 3), -deg2rad(360 * 2)])
        self.dangle_max = np.array([deg2rad(360 * 3), deg2rad(360 * 3), deg2rad(360 * 2)])
        'state limitation'

        '''datasave'''
        self.save_pos = np.atleast_2d([self.x, self.y, self.z])
        self.save_vel = np.atleast_2d([self.vx, self.vy, self.vz])
        self.save_angle = np.atleast_2d([self.phi, self.theta, self.psi])
        self.save_omega_body = np.atleast_2d([self.p, self.q, self.r])
        self.save_control_input = np.atleast_2d(self.control)
        self.save_t = np.array([self.time])
        '''datasave'''

        '''other'''
        # self.f1_rho1 = np.zeros((4, 4)).astype(float)
        # self.f2_rho2 = np.zeros((4, 4)).astype(float)
        # self.g_rho1 = np.zeros((4, 4)).astype(float)
        self.f1g_old: np.ndarray = np.dot(self.f1_rho1(), self.g_rho1())
        self.f1g_new: np.ndarray = np.dot(self.f1_rho1(), self.g_rho1())
        self.Frho2_f1f2_old: np.ndarray = np.dot(self.dot_f1_rho1(), self.rho2()) + np.dot(self.f1_rho1(), self.f2_rho2())
        self.Frho2_f1f2_new: np.ndarray = np.dot(self.dot_f1_rho1(), self.rho2()) + np.dot(self.f1_rho1(), self.f2_rho2())
        '''other'''

    def generate_uncertainty(self) -> np.ndarray:
        # Fdx, Fdy, Fdz, dp, dq, dr
        Fdx = 0.
        Fdy = 0.
        Fdz = 1 * np.sin(self.time)
        dp = 0.2 * np.sin(self.time)
        dq = 0.2 * np.sin(self.time)
        dr = 0.2 * np.sin(self.time)
        return np.array([Fdx, Fdy, Fdz, dp, dq, dr])

    def set_position_limitation2inf(self):
        """
        :func:      将无人机的位置限制设置为infinity，该函数仅在设计位置控制器时应用
        :return:
        """
        self.pos_max = np.array([np.inf, np.inf, np.inf])
        self.pos_min = np.array([-np.inf, -np.inf, -np.inf])

    def is_out(self):
        if max(np.fabs(self.phi), np.fabs(self.theta)) >= np.pi / 2:
            print('Attitude out...')
            return True
        if self.time > self.time_max:
            print('Time out')
            return True
        return False

    # def is_episode_Terminal(self):
    #     # self.terminal_flag = 0
    #     # if self.time > self.tmax:
    #     #     print('Time out...')
    #     #     self.terminal_flag = 2
    #     #     self.is_terminal = True
    #     #     return True
    #     #
    #     # if self.is_out():
    #     #     print('Out...')
    #     #     self.terminal_flag = 1
    #     #     self.is_terminal = True
    #     #     return True
    #
    #     self.is_terminal = False
    #     return False

    def get_ideal_w_rotor(self) -> np.ndarray:
        return np.sqrt(np.dot(np.linalg.inv(self.power_allocation_mat), self.control_ideal))

    def rotor_transfer(self) -> np.ndarray:
        return self.w_rotor_ideal

    def ode(self, xx: np.ndarray, dis: np.ndarray, is_motor_ideal: bool = True):
        """
        :param xx:              状态
        :param is_motor_ideal:  电机是否理想
        :return:                状态的导数
        """
        '''
        在微分方程里面的状态 X = [x y z vx vy vz phi theta psi p q r] 一共12个
        定义惯性系到机体系的旋转矩阵
        R_i_b1 = np.array([[math.cos(self.psi), math.sin(self.psi), 0],
                           [-math.sin(self.psi), math.cos(self.psi), 0],
                           [0, 0, 1]])  # 从惯性系到b1系，旋转偏航角psi
        R_b1_b2 = np.array([[math.cos(self.theta), 0, -math.sin(self.theta)],
                            [0, 1, 0],
                            [math.sin(self.theta), 0, math.cos(self.theta)]])  # 从b1系到b2系，旋转俯仰角theta
        R_b2_b = np.array([[1, 0, 0],
                           [0, math.cos(self.phi), math.sin(self.phi)],
                           [0, -math.sin(self.phi), math.cos(self.phi)]])  # 从b2系到b系，旋转滚转角phi
        R_i_b = np.matmul(R_b2_b, np.matmul(R_b1_b2, R_i_b1))  # 从惯性系到机体系的转换矩阵
        R_b_i = R_i_b.T  # 从机体系到惯性系的转换矩阵
        e3_i = np.array([0, 0, 1])  # 惯性系下的Z轴基向量
        # dx = v
        # dv = g*e3_i + f/m*
        '''
        [_x, _y, _z, _vx, _vy, _vz, _phi, _theta, _psi, _p, _q, _r] = xx[0:12]

        '''至此，已经计算出理想控制量的，油门 + 三转矩'''
        self.w_rotor_ideal = self.get_ideal_w_rotor()  # 计算出理想状态下电机的转速
        if is_motor_ideal:
            self.w_rotor = self.w_rotor_ideal.copy()
            self.control = self.control_ideal.copy()
        else:
            self.w_rotor = self.rotor_transfer()
            self.control = np.dot(self.power_allocation_mat, self.w_rotor ** 2)
        self.throttle = self.control[0]
        self.torque = self.control[1:4]

        '''至此，已经计算出实际控制量的，油门 + 三转矩'''

        # uncertainty = self.generate_uncertainty()  # 生成不确定性

        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''
        self.J0 = 0.        # 不考虑陀螺力矩
        dp = (-self.kr * _p - _q * _r * (self.J[2] - self.J[1]) + self.torque[0]
              - self.J0 * _q * (-self.w_rotor[0] - self.w_rotor[1] + self.w_rotor[2] + self.w_rotor[3])) / self.J[0] + dis[3]
        dq = (-self.kr * _q - _p * _r * (self.J[0] - self.J[2]) + self.torque[1]
              - self.J0 * _p * (self.w_rotor[0] + self.w_rotor[1] - self.w_rotor[2] - self.w_rotor[3])) / self.J[1] + dis[4]
        dr = (-self.kr * _r - _p * _q * (self.J[1] - self.J[0]) + self.torque[2]) / self.J[2] + dis[5]
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''

        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''
        _R_pqr2diner = np.array([[1, np.tan(_theta) * np.sin(_phi), np.tan(_theta) * np.cos(_phi)],
                                 [0, np.cos(_phi), -np.sin(_phi)],
                                 [0, np.sin(_phi) / np.cos(_theta), np.cos(_phi) / np.cos(_theta)]])
        [dphi, dtheta, dpsi] = np.dot(_R_pqr2diner, [_p, _q, _r]).tolist()
        '''2. 无人机在惯性系下的姿态角 phi theta psi 的微分方程'''

        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''
        [dx, dy, dz] = [_vx, _vy, _vz]
        dvx = (self.throttle * (np.cos(_psi) * np.sin(_theta) * np.cos(_phi) + np.sin(_psi) * np.sin(_phi))
               - self.kt * _vx + dis[0]) / self.m
        dvy = (self.throttle * (np.sin(_psi) * np.sin(_theta) * np.cos(_phi) - np.cos(_psi) * np.sin(_phi))
               - self.kt * _vy + dis[1]) / self.m
        dvz = -self.g + (self.throttle * np.cos(_phi) * np.cos(_theta)
                         - self.kt * _vz + dis[2]) / self.m
        '''3. 无人机在惯性系下的位置 x y z 和速度 vx vy vz 的微分方程'''

        return np.array([dx, dy, dz, dvx, dvy, dvz, dphi, dtheta, dpsi, dp, dq, dr])

    def rk44(self, action: np.ndarray, dis: np.ndarray, n: int = 10):
        self.control_ideal = action
        h = self.dt / n  # RK-44 解算步长
        # tt = self.time + self.dt

        '''时间更新之前，先把需要求导的保存'''
        self.f1g_old = self.f1g_new.copy()
        self.Frho2_f1f2_old = self.Frho2_f1f2_new.copy()
        '''时间更新之前，先把需要求导的保存'''
        cc = 0
        while cc < n:       # self.time < tt
            # xx_old = np.concatenate((self.pos, self.vel, self.angle, self.omega_body))
            xx_old = np.array([self.x, self.y, self.z,
                               self.vx, self.vy, self.vz,
                               self.phi, self.theta, self.psi,
                               self.p, self.q, self.r])
            K1 = h * self.ode(xx_old, dis)
            K2 = h * self.ode(xx_old + K1 / 2, dis)
            K3 = h * self.ode(xx_old + K2 / 2, dis)
            K4 = h * self.ode(xx_old + K3, dis)
            xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            xx_temp = xx_new.copy()
            [self.x, self.y, self.z] = xx_temp[0:3]
            [self.vx, self.vy, self.vz] = xx_temp[3:6]
            [self.phi, self.theta, self.psi] = xx_temp[6:9]
            [self.p, self.q, self.r] = xx_temp[9:12]
            cc += 1
        self.time += self.dt
        # R_pqr2diner = np.array([[1, np.tan(self.theta) * np.sin(self.phi), np.tan(self.theta) * np.cos(self.phi)],
        #                         [0, np.cos(self.phi), -np.sin(self.phi)],
        #                         [0, np.sin(self.phi) / np.cos(self.theta), np.cos(self.phi) / np.cos(self.theta)]])
        if self.psi > np.pi:  # 如果角度超过 180 度
            self.psi -= 2 * np.pi
        if self.psi < -np.pi:  # 如果角度小于 -180 度
            self.psi += 2 * np.pi
        self.n += 1  # 拍数 +1

        '''需要求导的变量更新'''
        self.f1g_new = np.dot(self.f1_rho1(), self.g_rho1())
        self.Frho2_f1f2_new = np.dot(self.dot_f1_rho1(), self.rho2()) + np.dot(self.f1_rho1(), self.f2_rho2())
        '''需要求导的变量更新'''

    def uav_state_call_back(self):
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r])

    def f1_rho1(self):
        """
        :brief:  [1  0          0                     0          ]
                 [0  1  sin(phi)tan(theta)            0          ]
                 [0  0       cos(phi)               -sin(phi)    ]
                 [0  0  sin(phi)/cos(theta)  -cos(phi)/cos(theta)]
        :return: f1(rho_1)
        """
        _f1_rho1 = np.zeros((4, 4)).astype(float)
        _f1_rho1[0][0] = 1.
        _f1_rho1[1][1] = 1.
        _f1_rho1[1][2] = np.sin(self.phi) * np.tan(self.theta)
        _f1_rho1[1][3] = np.cos(self.phi) * np.tan(self.theta)
        _f1_rho1[2][2] = np.cos(self.phi)
        _f1_rho1[2][3] = -np.sin(self.phi)
        _f1_rho1[3][2] = np.sin(self.phi) / np.cos(self.theta)
        _f1_rho1[3][3] = np.cos(self.phi) / np.cos(self.theta)
        return _f1_rho1

    def f2_rho2(self):
        """
        :brief:  [       -(mg + kt * dz) / m       ]
                 [(kr * p + qr * (Iyy - Izz)) / Ixx]
                 [(kr * q + pr * (Izz - Ixx)) / Iyy]
                 [(kr * r + pq * (Ixx - Iyy)) / Izz]
        :return: f2(rho_2)
        """
        _f2_rho2 = np.array([0, 0, 0, 0]).astype(float)
        _f2_rho2[0] = -(self.m * self.g + self.kt * self.vz) / self.m
        _f2_rho2[1] = (self.kr * self.p + self.q * self.r * (self.J[1] - self.J[2])) / self.J[0]
        _f2_rho2[2] = (self.kr * self.q + self.p * self.r * (self.J[2] - self.J[0])) / self.J[1]
        _f2_rho2[3] = (self.kr * self.r + self.p * self.q * (self.J[0] - self.J[1])) / self.J[2]
        return _f2_rho2

    def g_rho1(self):
        """
        :brief:  [1  0          0                     0          ]
                 [0  1  sin(phi)tan(theta)            0          ]
                 [0  0       cos(phi)               -sin(phi)    ]
                 [0  0  sin(phi)/cos(theta)  -cos(phi)/cos(theta)]
        :return: g(rho_1)
        """
        _g = np.zeros((4, 4)).astype(float)
        _g[0][0] = np.cos(self.phi) * np.cos(self.theta) / self.m
        _g[1][1] = 1 / self.J[0]
        _g[2][2] = 1 / self.J[1]
        _g[3][3] = 1 / self.J[2]
        return _g

    def rho1(self):
        return np.array([self.z, self.phi, self.theta, self.psi])

    def rho2(self):
        return np.array([self.vz, self.p, self.q, self.r])

    def dot_rho1(self):
        return np.dot(self.f1_rho1(), self.rho2())

    def dot_rho2(self):
        uncertainty = self.generate_uncertainty()
        # Fdx, Fdy, Fdz, dp, dq, dr
        Fdz, dp, dq, dr = uncertainty[2], uncertainty[3], uncertainty[4], uncertainty[5]
        return self.f2_rho2() + np.dot(self.g_rho1(), self.control) + np.array([Fdz / self.m, dp / self.J[0], dq / self.J[1], dr / self.J[2]])

    def dot_f1_rho1(self):
        dot_rho1 = self.dot_rho1()  # dz dphi dtheta dpsi
        _dot_f1_rho1 = np.zeros((4, 4)).astype(float)
        _dot_f1_rho1[1][2] = dot_rho1[1] * np.tan(self.theta) * np.cos(self.phi) + dot_rho1[2] * np.sin(self.phi) / np.cos(self.theta) ** 2
        _dot_f1_rho1[1][3] = -dot_rho1[1] * np.tan(self.theta) * np.sin(self.phi) + dot_rho1[2] * np.cos(self.phi) / np.cos(self.theta) ** 2

        _dot_f1_rho1[2][2] = -dot_rho1[1] * np.sin(self.phi)
        _dot_f1_rho1[2][3] = -dot_rho1[1] * np.cos(self.phi)

        _temp1 = dot_rho1[1] * np.cos(self.phi) * np.cos(self.theta) + dot_rho1[2] * np.sin(self.phi) * np.sin(self.theta)
        _dot_f1_rho1[3][2] = _temp1 / np.cos(self.theta) ** 2

        _temp2 = -dot_rho1[1] * np.sin(self.phi) * np.cos(self.theta) + dot_rho1[2] * np.cos(self.phi) * np.sin(self.theta)
        _dot_f1_rho1[3][3] = _temp2 / np.cos(self.theta) ** 2
        return _dot_f1_rho1

    def dot_f1g(self):
        return (self.f1g_new - self.f1g_old) / self.dt

    def dot_Frho2_f1f2(self):
        return (self.Frho2_f1f2_new - self.Frho2_f1f2_old) / self.dt
