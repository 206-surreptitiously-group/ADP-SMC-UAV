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
    testPath0 = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/position_tracking/nets/draw/'
    testPath_smc = None

    env = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
    reset_pos_ctrl_param('zero')
    env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)
    env.load_norm_normalizer_from_file(testPath0, 'state_norm.csv')
    actor = PPOActor_Gaussian(state_dim=env.state_dim,
                              action_dim=env.action_dim,
                              a_min=np.array(env.action_range)[:, 0],
                              a_max=np.array(env.action_range)[:, 1],
                              init_std=0.5,
                              use_orthogonal_init=True)
    actor.orthogonal_init_all()
    actor.load_state_dict(torch.load(testPath0 + 'actor'))
    agent = PPO(env=env, actor=actor)

    r = []
    att_out = []
    pos_out = []

    CREATE_FILE = False
    CONTROLLER = 'FNTSMC'

    if CREATE_FILE:
        param = []
        A = np.linspace(0, 1.5, 16)
        T = np.linspace(5, 10, 51)
        # phase_a = np.array([0., np.pi / 2, np.pi / 2])
        # bias_a = np.zeros(3)
        for _A in A:
            for _T in T:
                _param = [_A, _A, _A, 0, _T, _T, _T, _T, np.pi / 2, 0., 0., 0., 0., 0., 1.5, 0.]
                param.append(_param)

        (pd.DataFrame(param,
                      columns=['Ax', 'Ay', 'Az', 'Apsi', 'Tx', 'Ty', 'Tz', 'Tpsi', 'phi0x', 'phi0y', 'phi0z', 'phi0psi', 'bias_x', 'bias_y', 'bias_z', 'bias_psi']).
         to_csv('./uniform_comparative_pos/uniform_mc_test_traj.csv', sep=',', index=False))
    else:
        ref_trajectory = pd.read_csv('./uniform_comparative_pos/A_0-1.5_T_5-10_phi0_90-0-0-0_bias_0-0-1.5-0/uniform_mc_test_traj.csv', header=0).to_numpy()
        print(ref_trajectory.shape)
        cnt = 0
        for _traj in ref_trajectory:
            # 此文件仅仅测试 SMC 效果
            if cnt % 100 == 0:
                print('index: ', cnt)
            if CONTROLLER == 'FNTSMC':
                reset_pos_ctrl_param('optimal')
            else:
                reset_pos_ctrl_param('zero')
            A = _traj[0: 4]
            T = _traj[4: 8]
            # A = np.array([0.2,0.2,0.2,0])
            # T = np.ones(4) * 7
            phi0 = _traj[8: 12]
            env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False,
                                               random_pos0=False,
                                               new_att_ctrl_param=None,
                                               new_pos_ctrl_parma=pos_ctrl_param,
                                               outer_param=[A, T, phi0])
            test_r = 0.
            while not env.is_terminal:
                if CONTROLLER == 'RL':
                    a = agent.evaluate(env.current_state_norm(env.current_state, update=False))
                    env.get_param_from_actor(a)  # 将控制器参数更新
                a_4_uav = env.generate_action_4_uav()
                env.step_update(a_4_uav)
                test_r += env.reward

                # env.image = env.image_copy.copy()
                # env.draw_3d_points_projection(np.atleast_2d([env.uav_pos(), env.pos_ref]), [Color().Red, Color().DarkGreen])
                # env.draw_time_error(env.uav_pos(), env.pos_ref)
                # env.show_image(False)
            # print(test_r)
            r.append(test_r)
            att_out.append(1 if env.terminal_flag == 3 else 0)
            pos_out.append(1 if env.terminal_flag == 2 else 0)
            cnt += 1
        if CONTROLLER == 'RL':
            (pd.DataFrame({'r': r, 'att_out': att_out, 'pos_out': pos_out}).
             to_csv('./uniform_comparative_pos/uniform_mc_test_rl.csv', sep=',', index=False))
        else:
            (pd.DataFrame({'r': r, 'att_out': att_out, 'pos_out': pos_out}).
             to_csv('./uniform_comparative_pos/uniform_mc_test_smc.csv', sep=',', index=False))
