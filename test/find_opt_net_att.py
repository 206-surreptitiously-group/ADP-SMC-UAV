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
    env_test = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    reset_att_ctrl_param('zero')
    env_test.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)

    test_actor = PPOActor_Gaussian(state_dim=env_test.state_dim,
                                   action_dim=env_test.action_dim,
                                   a_min=np.array(env_test.action_range)[:, 0],
                                   a_max=np.array(env_test.action_range)[:, 1],
                                   init_std=0.5,
                                   use_orthogonal_init=True)
    testPath0 = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/att_maybe_optimal2/'
    index = 2350
    _step = 50
    test_r = 0.
    pos_out = 0
    att_out = 0
    n = 20

    # f = open("record_at.txt", 'w')
    while index <= 2350:
        print('==========test {}=========='.format(index))
        test_actor.orthogonal_init_all()
        testPath = testPath0 + 'trainNum_' + str(index) + '/'
        test_actor.load_state_dict(torch.load(testPath + 'actor_trainNum_' + str(index)))  # 测试时，填入测试actor网络
        agent = PPO(env=env_test, actor=test_actor, path='')
        env_test.load_norm_normalizer_from_file(testPath, 'state_norm.csv')

        index_sumr = 0.
        rr = []
        for i in range(n):
            reset_att_ctrl_param('zero')
            env_test.reset_uav_att_ctrl_RL_tracking(random_trajectroy=True,
                                                    yaw_fixed=False,
                                                    new_att_ctrl_param=att_ctrl_param,
                                                    outer_param=None)
            env_test.show_att_image(True)
            test_r = 0.
            att_out = 0
            while not env_test.is_terminal:
                _a = agent.evaluate(env_test.current_state_norm(env_test.current_state, update=False))
                _new_SMC_param = _a.copy()
                env_test.get_param_from_actor(_new_SMC_param)  # 将控制器参数更新
                _rhod, _dot_rhod, _, _ = ref_inner(env_test.time,
                                                   env_test.ref_att_amplitude,
                                                   env_test.ref_att_period,
                                                   env_test.ref_att_bias_a,
                                                   env_test.ref_att_bias_phase)
                _torque = env_test.att_control(_rhod, _dot_rhod, None)
                env_test.step_update([_torque[0], _torque[1], _torque[2]])
                test_r += env_test.reward

                env_test.att_image = env_test.att_image_copy.copy()
                env_test.draw_att(_rhod)
                env_test.show_att_image(iswait=False)
            if env_test.terminal_flag == 1:     # time out
                index_sumr += test_r
                rr.append(test_r)
            elif env_test.terminal_flag == 3:
                att_out += 1
            else:
                pass
            print(test_r)
            # if i % 50 == 0:
            #     print('i = ', i)
        # f.writelines('==== i %.0f: att_out: %.0f  | r = %.2f \n' % (index, 1 if env_test.terminal_flag == 3 else 0, test_r))
        # if att_out == 0:
        #     index_sumr /= (n - att_out)
        # else:
        #     pass
        # pd.DataFrame(rr, index=None).to_csv(testPath + 'test_sumr.csv', sep=',', index=False, header=False)
        # with open(testPath + "record.txt", 'w') as f:
        #     f.writelines('att_out: {}  success: {}'.format(att_out, n - att_out))

        # print('index: ', index, 'average reward: ', index_sumr, '\n')
        index += _step
    # f.close()
