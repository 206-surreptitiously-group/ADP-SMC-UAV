import numpy as np
from environment.UAV.uav import UAV
from environment.UAV.observer import inner_observer


class SMCControl(UAV):
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
                 pos0=None,
                 vel0=None,
                 angle0=None,
                 omega0_body=None):
        super(SMCControl, self).__init__(m, g, J, d, CT, CM, J0, kr, kt, pos0, vel0, angle0, omega0_body)
        self.inner_control = np.array([self.m * self.g, 0, 0, 0]).astype(float)  # 内环控制器，油门初始平衡重力，合理
        self.outer_control = np.array([0, 0]).astype(float)         # 外环控制器

        self.MODE = ['regulating', 'tracking']  # 调节问题 or 跟踪问题
        self.mode = self.MODE[0]                # 默认调节问题

        self.setPos = np.array([0, 0, 0]).astype(float)     # 用于调节问题的定点控制
        self.setTraj = np.zeros((1, 12), float)              # 用于跟踪问题的跟踪控制pos vel acc

        '''inner controller'''
        self.phi_d = [0., 0., 0.]
        self.theta_d = [0., 0., 0.]

        '''动态滑膜'''
        self.CI = np.array([3, 10, 10, 10]).astype(float)  # gain1 for switching function
        self.LambdaI = np.array([3, 10, 10, 10]).astype(float)    # gain for sliding mode function
        # self.K0_I = np.array([10, 15, 15, 25]).astype(float)    # 符号函数增益，理想
        self.K0_I = np.array([10, 15, 15, 25]).astype(float)  # 带有观测器
        '''动态滑膜'''

        '''传统滑膜'''
        # self.CI = np.array([5, 10, 10, 5]).astype(float)  # gain1 for switching function
        # self.LambdaI = np.array([5, 10, 10, 5]).astype(float)    # gain for sliding mode function
        # self.K0_I = np.array([1, 1, 1, 1]).astype(float)
        '''传统滑膜'''

        self.sI = np.array([0, 0, 0, 0]).astype(float)
        self.dsI = np.array([0, 0, 0, 0]).astype(float)
        self.sigmaI = np.array([0, 0, 0, 0]).astype(float)
        # self.sI_dsI_sigmaI_init()

        yd, _, _, _ = self.fake_inner_cmd_generator()

        _eI = self.rho1() - yd
        self.inner_obs = inner_observer(dt=self.dt, K=20., init_de=_eI)
        '''inner controller'''

        '''outer controller'''

        '''outer controller'''

    def sI_dsI_sigmaI_init(self):
        fake_rhod, fake_dot_rhod, fake_dotdot_rhod, fake_dotdotdot_rhod = self.fake_inner_cmd_generator()  # 参考信号肯定是已知的，这不用说，因为这是认为定义好的
        """
        :return:
        """
        # TODO 没有不确定性
        eI = self.rho1() - fake_rhod
        deI = self.dot_rho1() - fake_dot_rhod
        ddeI = np.dot(self.dot_f1_rho1(), self.rho2()) + np.dot(self.f1_rho1(), self.dot_rho2()) - fake_dotdot_rhod
        self.sI = self.CI * eI + deI
        self.dsI = self.CI * deI + ddeI
        self.sigmaI = self.dsI + self.LambdaI * self.sI

    def fake_inner_cmd_generator(self):
        """
        :return: 用于生成模拟的内环参考信号指令
        """
        Az = 1
        A0z = 2
        Tz = 4
        wz = 2 * np.pi / Tz

        _fake_z = Az * np.sin(wz * self.time) + A0z
        _fake_dz = Az * wz * np.cos(wz * self.time)
        _fake_ddz = -Az * wz ** 2 * np.sin(wz * self.time)
        _fake_dddz = -Az * wz ** 3 * np.cos(wz * self.time)

        T = 5
        w = 2 * np.pi / T
        A = np.pi / 6

        _fake_phi = A * np.sin(w * self.time)       # T = 4s A = pi/6 A0 = 0
        _fake_dphi = A * w * np.cos(w * self.time)
        _fake_ddphi = -A * w ** 2 * np.sin(w * self.time)
        _fake_dddphi = -A * w ** 3 * np.cos(w * self.time)

        _fake_theta = A * np.sin(w * self.time)  # T = 4s A = pi/6 A0 = 0
        _fake_dtheta = A * w * np.cos(w * self.time)
        _fake_ddtheta = -A * w ** 2 * np.sin(w * self.time)
        _fake_dddtheta = -A * w ** 3 * np.cos(w * self.time)

        T3 = 5
        w3 = 2 * np.pi / T3
        A3 = np.pi / 2
        _fake_psi = A3 * np.sin(w3 * self.time)  # T = 4s A = pi/6 A0 = 0
        _fake_dpsi = A3 * w3 * np.cos(w3 * self.time)
        _fake_ddpsi = -A3 * w3 ** 2 * np.sin(w3 * self.time)
        _fake_dddpsi = -A3 * w3 ** 3 * np.cos(w3 * self.time)

        # _fake_z = 3.
        # _fake_dz = _fake_ddz = _fake_dddz = 0.
        # _fake_phi = _fake_dphi = _fake_ddphi = _fake_dddphi = 0.
        # _fake_theta = _fake_dtheta = _fake_ddtheta = _fake_dddtheta = 0.
        # _fake_psi = _fake_dpsi = _fake_ddpsi = _fake_dddpsi = 0.

        return \
            np.array([_fake_z, _fake_phi, _fake_theta, _fake_psi]), \
            np.array([_fake_dz, _fake_dphi, _fake_dtheta, _fake_dpsi]), \
            np.array([_fake_ddz, _fake_ddphi, _fake_ddtheta, _fake_ddpsi]), np.array([_fake_dddz, _fake_dddphi, _fake_dddtheta, _fake_dddpsi])
