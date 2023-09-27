import numpy as np


class uav:
    def __init__(self,
                 m: float = 0.8,
                 g: float = 9.8,
                 J: np.ndarray = np.array([4.212e-3, 4.212e-3, 8.255e-3]),
                 d: float = 0.12,
                 CT: float = 2.168e-6,
                 CM: float = 2.136e-8,
                 J0: float = 1.01e-5,
                 kr: float = 1e-3,
                 kt: float = 1e-3,
                 pos0: np.ndarray = np.array([0, 0, 0]),
                 vel0: np.ndarray = np.array([0, 0, 0]),
                 angle0: np.ndarray = np.array([0, 0, 0]),
                 omega0_body: np.ndarray = np.array([0, 0, 0])
                 ):
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

        self.dt = 0.01
        self.n = 0  # 记录走过的拍数
        self.time = 0.  # 当前时间
        self.time_max = 30  # 每回合最大时间

        self.throttle = self.m * self.g  # 油门
        self.torque = np.array([0., 0., 0.]).astype(float)  # 转矩

    def ode(self, xx: np.ndarray, dis: np.ndarray):
        """
        @param xx:      state of the uav
        @param dis:     disturbances
        @return:        dot_xx
        """
        [_x, _y, _z, _vx, _vy, _vz, _phi, _theta, _psi, _p, _q, _r] = xx[0:12]
        '''1. 无人机绕机体系旋转的角速度p q r 的微分方程'''
        self.J0 = 0.  # 不考虑陀螺力矩，用于分析观测器的效果
        dp = (-self.kr * _p - _q * _r * (self.J[2] - self.J[1]) + self.torque[0]) / self.J[0] + dis[3]
        dq = (-self.kr * _q - _p * _r * (self.J[0] - self.J[2]) + self.torque[1]) / self.J[1] + dis[4]
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
        self.throttle = action[3]
        self.torque = action[0: 3]
        h = self.dt / n  # RK-44 解算步长
        cc = 0
        while cc < n:  # self.time < tt
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
        if self.psi > np.pi:  # 如果角度超过 180 度
            self.psi -= 2 * np.pi
        if self.psi < -np.pi:  # 如果角度小于 -180 度
            self.psi += 2 * np.pi
        self.n += 1  # 拍数 +1

    def uav_state_call_back(self):
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.phi, self.theta, self.psi, self.p, self.q, self.r])

    def uav_pos_vel_call_back(self):
        return np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz])

    def uav_att_pqr_call_back(self):
        return np.array([self.phi, self.theta, self.psi, self.p, self.q, self.r])

    def uav_pos(self):
        return np.array([self.x, self.y, self.z])

    def uav_vel(self):
        return np.array([self.vx, self.vy, self.vz])

    def uav_att(self):
        return np.array([self.phi, self.theta, self.psi])

    def uav_pqr(self):
        return np.array([self.p, self.q, self.r])
