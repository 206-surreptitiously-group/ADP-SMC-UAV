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

from environment.envs.UAV.uav_pos_ctrl_RL import uav_pos_ctrl_RL, uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
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

'''Parameter list of the position controller'''
pos_ctrl_param = fntsmc_param()

'''RL 学习初始参数'''
pos_ctrl_param.k1 = np.array([0., 0., 0.])  # 1.2, 0.8, 0.5
pos_ctrl_param.k2 = np.array([0., 0., 0.])  # 0.2, 0.6, 0.5
pos_ctrl_param.alpha = np.array([1.2, 1.5, 1.2])
pos_ctrl_param.beta = np.array([0.3, 0.3, 0.5])
pos_ctrl_param.gamma = np.array([0., 0., 0.])  # 0.2
pos_ctrl_param.lmd = np.array([0., 0., 0.])  # 2.0
'''RL 学习初始参数'''

'''传统控制参数'''
# pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
# pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
# pos_ctrl_param.alpha = np.array([1.2, 1.5, 1.2])
# pos_ctrl_param.beta = np.array([0.3, 0.3, 0.5])
# pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
# pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
'''传统控制参数'''

pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
'''Parameter list of the position controller'''

test_episode = []
test_reward = []
sumr_list = []


def reset_pos_ctrl_param(flag: str):
    if flag == 'zero':
        pos_ctrl_param.k1 = 0.01 * np.ones(3)
        pos_ctrl_param.k2 = 0.01 * np.ones(3)
        pos_ctrl_param.gamma = 0.01 * np.ones(3)
        pos_ctrl_param.lmd = 0.01 * np.ones(3)
    elif flag == 'random':
        pos_ctrl_param.k1 = np.random.random(3)
        pos_ctrl_param.k2 = np.random.random(3)
        pos_ctrl_param.gamma = np.random.random() * np.ones(3)
        pos_ctrl_param.lmd = np.random.random() * np.ones(3)
    else:  # optimal
        pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
        pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
        pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
        pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])


if __name__ == '__main__':
    # log_dir = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/log/'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # simulationPath = log_dir + 'test' + '-' + ALGORITHM + '-' + ENV + '/'
    # if not os.path.exists(log_dir):
    #     os.mkdir(simulationPath)
    env_test = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
    reset_pos_ctrl_param('zero')
    env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True, random_pos0=True, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)

    test_actor = PPOActor_Gaussian(state_dim=env_test.state_dim,
                                   action_dim=env_test.action_dim,
                                   a_min=np.array(env_test.action_range)[:, 0],
                                   a_max=np.array(env_test.action_range)[:, 1],
                                   init_std=0.5,
                                   use_orthogonal_init=True)
    testPath0 = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/maybe_optimal1/'
    index = 50
    _step = 50
    test_r = 0.
    pos_out = 0
    att_out = 0
    n = 500

    while index <= 5850:
        print('==========test {}=========='.format(index))
        test_actor.orthogonal_init_all()
        testPath = testPath0 + 'trainNum_' + str(index) + '/'
        test_actor.load_state_dict(torch.load(testPath + 'actor_trainNum_' + str(index)))  # 测试时，填入测试actor网络
        agent = PPO(env=env_test, actor=test_actor, path='')
        env_test.load_norm_normalizer_from_file(testPath, 'state_norm.csv')

        index_sumr = 0.
        rr = []

        for i in range(n):
            reset_pos_ctrl_param('zero')
            env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True,
                                                    random_pos0=True,
                                                    new_att_ctrl_param=None,
                                                    new_pos_ctrl_parma=pos_ctrl_param)
            # env_test.show_image(False)
            test_r = 0.
            pos_out = 0
            att_out = 0
            while not env_test.is_terminal:
                _a = agent.evaluate(env_test.current_state_norm(env_test.current_state, update=False))
                _new_SMC_param = _a.copy()
                env_test.get_param_from_actor(_new_SMC_param)  # 将控制器参数更新
                _a_4_uav = env_test.generate_action_4_uav()
                env_test.step_update(_a_4_uav)
                test_r += env_test.reward
            if env_test.terminal_flag == 1:     # time out
                index_sumr += test_r
                rr.append(test_r)
            elif env_test.terminal_flag == 2:
                pos_out += 1
            elif env_test.terminal_flag == 3:
                att_out += 1
            else:
                pass
        index_sumr /= (n - pos_out - att_out)
        pd.DataFrame(rr, index=None).to_csv(testPath + 'test_sumr.csv', sep=',', index=False, header=False)
        with open(testPath + "record.txt", 'w') as f:
            f.writelines('pos_out: {}  att_out: {}  success: {}'.format(pos_out, att_out, n - pos_out - att_out))
        print('index: ', index, 'average reward: ', index_sumr, '\n')
        index += _step
