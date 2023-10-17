from common.common_func import *
from algorithm.rl_base.rl_base import rl_base
import cv2 as cv
from environment.Color import Color


class Flight_Attitude_Simulator_Continuous(rl_base):
    def __init__(self, initTheta: float, setTheta: float):
        """
        @note:                  initialization
        @param initTheta:       initial theta
        @param setTheta:        target theta=0
        """
        super(Flight_Attitude_Simulator_Continuous, self).__init__()
        '''physical parameters'''
        self.name = 'Flight_Attitude_Simulator_Continuous2'
        self.setTheta = deg2rad(setTheta)
        self.force = 0
        self.f_max = 3.5
        self.f_min = -1.5

        self.minTheta = deg2rad(-60.0)
        self.maxTheta = deg2rad(60.0)

        self.min_omega = deg2rad(-90)
        self.max_omega = deg2rad(90)

        self.theta = deg2rad(initTheta)
        self.dTheta = 0.0

        self.dt = 0.02  # control period
        self.time = 0.0

        self.timeMax = 5

        self.Lw = 0.02  # 杆宽度
        self.L = 0.362  # 杆半长
        self.J = 0.082  # 转动惯量
        self.k = 0.09  # 摩擦系数
        self.m = 0.3  # 配重重量
        self.dis = 0.3  # 铜块中心距中心距离0.059
        self.copperl = 0.06  # 铜块长度
        self.copperw = 0.03  # 铜块宽度
        self.g = 9.8  # 重力加速度
        '''physical parameters'''

        '''RL_BASE'''
        # 这个状态与控制系统的状态不一样
        self.staticGain = 2
        self.state_dim = 2  # Theta, dTheta
        self.state_num = [math.inf for _ in range(self.state_dim)]
        self.state_step = [None for _ in range(self.state_dim)]
        self.state_space = [None for _ in range(self.state_dim)]
        self.state_range = [[-self.staticGain, self.staticGain] for _ in range(self.state_dim)]
        self.isStateContinuous = [True for _ in range(self.state_dim)]
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()

        self.action_dim = 1
        self.action_step = [None]
        self.action_range = [[self.f_min, self.f_max]]
        self.action_num = [math.inf]
        self.action_space = [None]
        self.isActionContinuous = True
        self.initial_action = np.array([0.0])
        self.current_action = self.initial_action.copy()

        self.reward = 0.0
        self.Q = 1
        self.Qv = 0.
        self.R = 0.1
        self.terminal_flag = 0  # 0-正常 1-上边界出界 2-下边界出界 3-超时
        self.is_terminal = False
        '''RL_BASE'''

        '''visualization_opencv'''
        self.width = 400
        self.height = 400
        self.image = np.zeros([self.width, self.height, 3], np.uint8)
        self.image[:, :, 0] = np.ones([self.width, self.height]) * 255
        self.image[:, :, 1] = np.ones([self.width, self.height]) * 255
        self.image[:, :, 2] = np.ones([self.width, self.height]) * 255
        self.name4image = 'Flight attitude simulator'
        self.scale = 250  # cm -> pixel
        self.ybias = 360  # pixel
        self.base_hor_w = 0.4
        self.base_hor_h = 0.02
        self.base_ver_w = 0.02
        self.base_ver_h = 0.8

        self.show = self.image.copy()
        self.save = self.image.copy()

        self.draw_base()
        self.draw_pendulum()
        self.draw_copper()
        # self.show_initial_image(isWait=True)
        '''visualization_opencv'''

        '''data_save'''
        self.save_Time = [self.time]
        self.save_Theta = [self.theta]
        self.save_dTheta = [self.dTheta]
        self.save_F = [self.initial_action[0]]
        '''data_save'''

    def draw_base(self):
        """
        :brief:     绘制基座
        :return:    None
        """
        pt1 = (int(self.width / 2 - self.base_hor_w * self.scale / 2), self.ybias)
        pt2 = (int(pt1[0] + self.base_hor_w * self.scale), int(pt1[1] - self.base_hor_h * self.scale))
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=-1)
        pt1 = (int(self.width / 2 - self.base_ver_w * self.scale / 2), pt2[1])
        pt2 = (int(pt1[0] + self.base_ver_w * self.scale), int(pt2[1] - self.base_ver_h * self.scale))
        cv.rectangle(self.image, pt1=pt1, pt2=pt2, color=Color().Blue, thickness=-1)
        self.show = self.image.copy()

    def draw_pendulum(self):
        """
        :brief:     绘制摆杆
        :return:    None
        """
        cx = int(self.width / 2)
        cy = int(self.ybias - (self.base_hor_h + self.base_ver_h) * self.scale)
        theta1 = np.arctan(self.Lw / self.L / 2)
        theta2 = -theta1
        theta3 = math.pi + theta1
        theta4 = math.pi + theta2
        L0 = np.sqrt((self.Lw / 2) ** 2 + self.L ** 2)
        pt1 = np.atleast_1d([int(L0 * np.cos(theta1 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta1 + self.theta) * self.scale)])
        pt2 = np.atleast_1d([int(L0 * np.cos(theta2 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta2 + self.theta) * self.scale)])
        pt3 = np.atleast_1d([int(L0 * np.cos(theta3 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta3 + self.theta) * self.scale)])
        pt4 = np.atleast_1d([int(L0 * np.cos(theta4 + self.theta) * self.scale + cx),
                             int(cy - L0 * np.sin(theta4 + self.theta) * self.scale)])
        cv.fillPoly(img=self.show, pts=np.array([[pt1, pt2, pt3, pt4]]), color=Color().Red)
        # self.show = self.image.copy()

    def draw_copper(self):
        """
        :brief:     绘制铜块
        :return:    None
        """
        cx = int(self.width / 2)
        cy = int(self.ybias - (self.base_hor_h + self.base_ver_h) * self.scale)
        theta1 = np.arctan(self.copperw / 2 / (self.dis - self.copperl / 2))
        theta2 = np.arctan(self.copperw / 2 / (self.dis + self.copperl / 2))
        theta3 = -theta2
        theta4 = -theta1

        l1 = np.sqrt((self.copperw / 2) ** 2 + (self.dis - self.copperl / 2) ** 2)
        l2 = np.sqrt((self.copperw / 2) ** 2 + (self.dis + self.copperl / 2) ** 2)

        pt1 = np.atleast_1d([int(l1 * np.cos(theta1 + self.theta) * self.scale + cx),
                             int(cy - l1 * np.sin(theta1 + self.theta) * self.scale)])
        pt2 = np.atleast_1d([int(l2 * np.cos(theta2 + self.theta) * self.scale + cx),
                             int(cy - l2 * np.sin(theta2 + self.theta) * self.scale)])
        pt3 = np.atleast_1d([int(l2 * np.cos(theta3 + self.theta) * self.scale + cx),
                             int(cy - l2 * np.sin(theta3 + self.theta) * self.scale)])
        pt4 = np.atleast_1d([int(l1 * np.cos(theta4 + self.theta) * self.scale + cx),
                             int(cy - l1 * np.sin(theta4 + self.theta) * self.scale)])

        cv.fillPoly(img=self.show, pts=np.array([[pt1, pt2, pt3, pt4]]), color=Color().Black)
        # self.show = self.image.copy()

    def draw_text(self):
        _s = 'Theta: %.2f' % (rad2deg(self.theta))
        cv.putText(self.show, _s, (100, 100), cv.FONT_HERSHEY_COMPLEX, 0.6, Color().Black, 2)

    def show_initial_image(self, isWait):
        cv.imshow(self.name4image, self.show)
        cv.waitKey(0) if isWait else cv.waitKey(1)

    def show_dynamic_image(self, isWait=False):
        self.draw_pendulum()
        self.draw_copper()
        self.draw_text()
        cv.imshow(self.name4image, self.show)
        cv.waitKey(0) if isWait else cv.waitKey(1)
        self.save = self.show.copy()
        self.show = self.image.copy()

    def state_norm(self) -> np.ndarray:
        """
        状态归一化
        """
        # # initTheta, Theta, dTheta, error
        _Theta = (2 * self.theta - self.maxTheta - self.minTheta) / (self.maxTheta - self.minTheta) * self.staticGain
        _dTheta = (2 * self.dTheta - self.max_omega - self.min_omega) / (self.max_omega - self.min_omega) * self.staticGain
        norm_state = np.array([_Theta, _dTheta])

        return norm_state

    def inverse_state_norm(self, s: np.ndarray) -> np.ndarray:
        _Theta = (s[0] / self.staticGain * (self.maxTheta - self.minTheta) + self.maxTheta + self.minTheta) / 2
        _dTheta = (s[1] / self.staticGain * (self.max_omega - self.min_omega) + self.max_omega + self.min_omega) / 2
        inv_norm_state = np.array([_Theta, _dTheta])

        return inv_norm_state

    def is_success(self):
        if np.fabs(self.theta) < deg2rad(1):       # 角度误差小于1度
            if np.fabs(self.dTheta) < deg2rad(1):       # 速度也很小
                return True
        return False

    def is_Terminal(self, param=None):
        """
        :brief:     判断回合是否结束
        :return:    是否结束
        """
        if self.theta > self.maxTheta + deg2rad(1):
            self.terminal_flag = 1
            # print('超出最大角度')
            return True
        if self.theta < self.minTheta - deg2rad(1):
            self.terminal_flag = 2
            # print('超出最小角度')
            return True
        if self.time > self.timeMax:
            self.terminal_flag = 3
            print('Timeout')
            return True
        # if self.is_success():
        #     self.terminal_flag = 4
        #     print('Success')
        #     return True
        self.terminal_flag = 0
        return False

    def get_reward(self, param=None):
        # s = self.inverse_state_norm(self.current_state)
        # ss = self.inverse_state_norm(self.next_state)
        # current_error = math.fabs(s[3])
        # next_error = math.fabs(ss[3])
        #
        # new_theta = ss[1]
        # new_dtheta = ss[2]

        # if current_error > next_error:  # 如果误差变小
        #     r1 = 3  # + (current_error - next_error) * 20
        # elif current_error < next_error:
        #     r1 = -3  # + (current_error - next_error) * 20
        # else:
        #     if next_error > deg2rad(0.5):  # 如果当前误差大于0.5度
        #         r1 = -3
        #     else:
        #         r1 = 5
        #
        # if self.terminal_flag == 2 or self.terminal_flag == 3:
        #     r2 = -5
        # elif self.terminal_flag == 4:
        #     r2 = 10
        # else:
        #     r2 = 0
        #
        # if new_theta * new_dtheta < 0:
        #     r3 = 5
        # else:
        #     r3 = -5
        Q = 3.
        R = 0.0
        r1 = -self.theta ** 2 * Q
        r2 = -self.dTheta ** 2 * R
        if self.terminal_flag == 1 or self.terminal_flag == 2:      # 出界
            r3 = (self.timeMax - self.time) / self.dt * r1
        elif self.terminal_flag == 4:   # 成功
            r3 = 0.
        else:
            r3 = 0.
        # if r1 + r2 + r3 > 0:
        #     print('日了')
        self.reward = r1 + r2 + r3


    # def get_reward(self, param=None):
    #     c_s = self.inverse_state_norm(self.current_state)
    #     n_s = self.inverse_state_norm(self.next_state)
    #
    #     abs_cur_t = math.fabs(c_s[0])
    #     abs_nex_t = math.fabs(n_s[0])
    #
    #     '''引导误差'''
    #     if abs_cur_t > abs_nex_t:  # 如果误差变小
    #         r1 = 2
    #     elif abs_cur_t < abs_nex_t:
    #         r1 = -2
    #     else:
    #         r1 = 0
    #     '''引导误差'''
    #
    #     '''引导方向'''
    #     if n_s[0] * n_s[1] < 0:
    #         r2 = 1
    #     else:
    #         r2 = -1
    #     '''引导方向'''
    #
    #     if self.terminal_flag == 1 or self.terminal_flag == 2:  # 出界
    #         r3 = -100
    #     elif self.terminal_flag == 3:   # 超市
    #         r3 = -0
    #     elif self.terminal_flag == 4:   # 成功
    #         r3 = 500
    #     else:
    #         r3 = 0
    #
    #     if math.fabs(self.theta) < deg2rad(1):
    #         r4 = 2
    #         if math.fabs(self.dTheta) < deg2rad(5):
    #             r4 += 2
    #     else:
    #         r4 = 0
    #     # r3 = 0
    #
    #     # r1 = -(next_error ** 2) * 100     # 使用 x'Qx 的形式，试试好不好使
    #     # r2 = -(self.dTheta ** 2) * 1
    #     # r3 = 0
    #
    #     self.reward = r1 + r2 + r3 + r4

    def ode(self, xx: np.ndarray):
        _dtheta = xx[1]
        _ddtheta = (self.force * self.L - self.m * self.g * self.dis - self.k * xx[1]) / (self.J + self.m * self.dis ** 2)
        return np.array([_dtheta, _ddtheta])

    def rk44(self, action: float):
        self.force = action

        xx_old = np.array([self.theta, self.dTheta])
        K1 = self.dt * self.ode(xx_old)
        K2 = self.dt * self.ode(xx_old + K1 / 2)
        K3 = self.dt * self.ode(xx_old + K2 / 2)
        K4 = self.dt * self.ode(xx_old + K3)
        xx_new = xx_old + (K1 + 2 * K2 + 2 * K3 + K4) / 6
        # xx_temp = xx_new.copy()
        self.theta = xx_new[0]
        self.dTheta = xx_new[1]
        self.time += self.dt

    def step_update(self, action: np.ndarray):
        self.current_action = action.copy()
        self.current_state = self.state_norm()  # 当前状态

        '''rk44'''
        self.rk44(action=action[0])
        # print('角速度',rad2deg(self.dTheta))
        '''rk44'''

        self.is_terminal = self.is_Terminal()
        self.next_state = self.state_norm()  # 下一时刻

        self.get_reward()

        # '''出界处理'''
        # if self.theta > self.maxTheta:                  # 如果超出最大角度限制
        #     self.theta = self.maxTheta
        #     self.dTheta = -0.8 * self.dTheta            # 碰边界速度直接反弹
        # if self.theta < self.minTheta:
        #     self.theta = self.minTheta
        #     self.dTheta = -0.8 * self.dTheta
        # '''出界处理'''

    def reset(self):
        """
        :brief:     reset
        :return:    None
        """
        '''physical parameters'''
        self.theta = 0.0
        self.dTheta = 0.0
        self.time = 0.0
        self.draw_base()
        self.draw_pendulum()
        self.draw_copper()
        '''physical parameters'''

        '''RL_BASE'''
        self.current_state = self.state_norm()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        '''RL_BASE'''

        '''data_save'''
        self.save_Time = [self.time]
        self.save_Theta = [self.theta]
        self.save_dTheta = [self.dTheta]
        self.save_F = [self.initial_action[0]]
        '''data_save'''

    def reset_random(self):
        """
        :brief:
        :return:
        """
        '''physical parameters'''
        self.theta = random.uniform(self.minTheta + deg2rad(10), self.maxTheta - deg2rad(10))
        self.dTheta = 0.0
        self.time = 0.0
        self.draw_base()
        self.draw_pendulum()
        self.draw_copper()
        # self.show_initial_image(isWait=True)
        '''physical parameters'''

        '''RL_BASE'''
        # 这个状态与控制系统的状态不一样
        self.initial_state = self.state_norm()
        self.current_state = self.initial_state.copy()
        self.next_state = self.initial_state.copy()
        self.current_action = self.initial_action.copy()
        self.reward = 0.0
        self.is_terminal = False
        self.terminal_flag = 0
        '''RL_BASE'''

        '''data_save'''
        self.save_Time = [self.time]
        self.save_Theta = [self.theta]
        self.save_dTheta = [self.dTheta]
        self.save_F = [self.initial_action[0]]
        '''data_save'''
