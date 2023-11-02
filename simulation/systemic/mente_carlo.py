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
from environment.envs.UAV.uav_att_ctrl_RL import uav_att_ctrl_RL
from environment.envs.UAV.FNTSMC import fntsmc_param
from environment.Color import Color
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from algorithm.policy_base.Proximal_Policy_Optimization import PPOActor_Gaussian, PPOCritic
from common.common_cls import *
from common.common_func import *

'''Parameter list of the quadrotor'''
DT = 0.01
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
pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
pos_ctrl_param.alpha = np.array([1.2, 1.5, 1.2])
pos_ctrl_param.beta = np.array([0.3, 0.3, 0.5])
pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])
pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
'''Parameter list of the position controller'''

test_episode = []
test_reward = []
r_pl = []
r_al = []


def plot_SMC_param(param: np.ndarray, msg=None):
    param = np.delete(param, 0, axis=0)
    xx = np.arange(param.shape[0])
    plt.figure()
    plt.grid(True)
    plt.plot(xx, param[:, 0], label='k11')
    plt.plot(xx, param[:, 1], label='k12')
    plt.plot(xx, param[:, 2], label='k13')
    plt.plot(xx, param[:, 3], label='k21')
    plt.plot(xx, param[:, 4], label='k22')
    plt.plot(xx, param[:, 5], label='k23')
    plt.plot(xx, param[:, 6], label='gamma')
    plt.plot(xx, param[:, 7], label='lambda')
    if msg:
        plt.title(msg)
    plt.legend()


def get_normalizer_from_file(dim, path, file):
    norm = Normalization(dim)
    data = pd.read_csv(path + file, header=0).to_numpy()
    norm.running_ms.n = data[0, 0]
    norm.running_ms.mean = data[:, 1]
    norm.running_ms.std = data[:, 2]
    norm.running_ms.S = data[:, 3]
    norm.running_ms.n = data[0, 4]
    norm.running_ms.mean = data[:, 5]
    norm.running_ms.std = data[:, 6]
    norm.running_ms.S = data[:, 7]
    return norm


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
    curPath = os.path.dirname(os.path.abspath(__file__))
    reset_pos_ctrl_param('optimal')
    reset_att_ctrl_param('optimal')

    env_pos = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
    env_pos.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)

    env_att = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    env_att.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False, yaw_fixed=False, new_att_ctrl_param=att_ctrl_param, outer_param=None)
    env_pos_msg = {'state_dim': env_pos.state_dim, 'action_dim': env_pos.action_dim, 'name': env_pos.name, 'action_range': env_pos.action_range}
    env_att_msg = {'state_dim': env_att.state_dim, 'action_dim': env_att.action_dim, 'name': env_att.name, 'action_range': env_att.action_range}

    a_att_cof = np.array([10., 10., 10., 0.1, 0.1, 0.1, 1., 1.])

    opt_pos = PPOActor_Gaussian(state_dim=env_pos.state_dim,
                                action_dim=env_pos.action_dim)
    optPathPos = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/position_tracking/nets/pos_new1-260/'
    opt_pos.load_state_dict(torch.load(optPathPos + 'actor'))
    pos_norm = get_normalizer_from_file(env_pos.state_dim, optPathPos, 'state_norm.csv')

    opt_att = PPOActor_Gaussian(state_dim=env_att.state_dim,
                                action_dim=env_att.action_dim)
    optPathAtt = os.path.dirname(os.path.abspath(__file__)) + '/../../datasave/nets/attitude_tracking/nets/draw_and_opt/'
    opt_att.load_state_dict(torch.load(optPathAtt + 'actor'))
    att_norm = get_normalizer_from_file(env_att.state_dim, optPathAtt, 'state_norm.csv')

    cnt = 0
    r = []
    att_out = []
    pos_out = []
    TEST_NUM = 2

    for i in range(TEST_NUM):
        # writer = cv.VideoWriter(curPath+'/record%.0f.mp4' % (cnt), cv.VideoWriter_fourcc(*'mp4v'), 250, (env_pos.width, env_pos.height))
        if cnt % 100 == 0:
            print('index: ', cnt)
        # A = np.random.uniform(0, 1, 4) * np.array([1.5, 1.5, 1.5, np.pi / 2])
        # T = np.random.uniform(5, 10, 4)
        # phi0 = np.random.uniform(0, np.pi / 2, 4)

        A = np.array([1.5, 1.5, 1.5, 0])
        T = np.array([5, 5, 5, 5])
        phi0 = np.array([np.pi / 2, 0, 0., 0])

        env_pos.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False,
                                               random_pos0=False,
                                               new_att_ctrl_param=None,
                                               new_pos_ctrl_parma=pos_ctrl_param,
                                               outer_param=[A, T, phi0])
        test_r = 0.
        while not env_pos.is_terminal:
            CONTROLLER = 'RL'
            # CONTROLLER = 'fntsmc'
            if CONTROLLER == 'RL':
                e_pos = env_pos.uav_pos() - env_pos.pos_ref
                e_vel = env_pos.uav_vel() - env_pos.dot_pos_ref
                pos_s = np.concatenate((e_pos, e_vel))
                param_pos = opt_pos.evaluate(pos_norm(pos_s))   # new position control parameter

                e_att = env_pos.uav_att() - env_pos.att_ref
                e_dot_att = env_pos.uav_dot_att() - env_pos.dot_att_ref
                att_s = np.concatenate((e_att, e_dot_att))
                parma_att = opt_att.evaluate(att_norm(att_s))   # new attitude control parameter

                env_pos.get_param_from_actor(param_pos, update_k2=False)     # update position control parameter
                env_pos.set_att_ctrl_from_outer(parma_att * a_att_cof)      # update attitude control parameter

            DRAW = True
            if DRAW:
                env_pos.image = env_pos.image_copy.copy()
                env_pos.draw_3d_points_projection(np.atleast_2d([env_pos.uav_pos(), env_pos.pos_ref]), [Color().Red, Color().DarkGreen])
                env_pos.draw_time_error(env_pos.uav_pos(), env_pos.pos_ref)
                env_pos.show_image(False)

            # RECORD = False
            # if RECORD:
            #     env_pos.image = env_pos.image_copy.copy()
            #     env_pos.draw_3d_points_projection(np.atleast_2d([env_pos.uav_pos(), env_pos.pos_ref]), [Color().Red, Color().DarkGreen])
            #     env_pos.draw_time_error(env_pos.uav_pos(), env_pos.pos_ref)
            #     # writer.write(env_pos.image)

            a_4_uav = env_pos.generate_action_4_uav(use_observer=True, is_ideal=False)
            env_pos.step_update(a_4_uav)
            test_r += env_pos.reward
        # writer.release()
        test_r *= DT
        r.append(test_r)
        print(test_r)
        att_out.append(1 if env_pos.terminal_flag == 3 else 0)
        pos_out.append(1 if env_pos.terminal_flag == 2 else 0)
        cnt += 1

        env_pos.collector.plot_outer_obs()
        env_pos.collector.plot_pos()
        plt.show()

    # if CONTROLLER == 'RL':
    #     (pd.DataFrame({'r': r, 'att_out': att_out, 'pos_out': pos_out}).
    #      to_csv('./mento_carlo_result_rl.csv', sep=',', index=False))
    # else:
    #     (pd.DataFrame({'r': r, 'att_out': att_out, 'pos_out': pos_out}).
    #      to_csv('./mento_carlo_result_smc.csv', sep=',', index=False))
    print('Finish simulation. | Pos_out: %.0f | Att_out: %.0f' % (np.sum(pos_out), np.sum(att_out)))
