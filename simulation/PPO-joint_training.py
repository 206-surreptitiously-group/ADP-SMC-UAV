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


timestep = 0
ENV = 'joint-training'
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
    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)

    TRAIN = True
    RETRAIN = True
    TEST = not TRAIN

    '''定义环境'''
    reset_pos_ctrl_param('optimal')
    reset_att_ctrl_param('optimal')

    env_pos = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
    env_pos.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)
    # env_pos.show_image(True)

    env_att = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    env_att.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False,  yaw_fixed=False, new_att_ctrl_param=att_ctrl_param, outer_param=None)

    env_test = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
    env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)
    '''定义环境'''

    reward_pos_norm = Normalization(shape=1)
    reward_att_norm = Normalization(shape=1)

    env_pos_msg = {'state_dim': env_pos.state_dim, 'action_dim': env_pos.action_dim, 'name': env_pos.name, 'action_range': env_pos.action_range}
    env_att_msg = {'state_dim': env_att.state_dim, 'action_dim': env_att.action_dim, 'name': env_att.name, 'action_range': env_att.action_range}

    a_att_cof = np.array([10., 10., 10., 0.1, 0.1, 0.1, 1., 1.])

    if TRAIN:
        action_std_init = 0.4  # 初始探索方差
        min_action_std = 0.05  # 最小探索方差
        max_train_steps = int(5e6)
        buffer_size = int(env_pos.time_max / env_pos.dt) * 2
        # K_epochs = 50
        t_epoch = 0  # 当前训练次数
        test_num = 0
        agent_pos = PPO(env_msg=env_pos_msg,
                        actor_lr=5e-5,
                        critic_lr=5e-4,
                        gamma=0.99,
                        K_epochs=25,
                        eps_clip=0.2,
                        action_std_init=action_std_init,
                        buffer_size=buffer_size,
                        max_train_steps=max_train_steps,
                        actor=PPOActor_Gaussian(state_dim=env_pos.state_dim,
                                                action_dim=env_pos.action_dim,
                                                a_min=np.array(env_pos.action_range)[:, 0],
                                                a_max=np.array(env_pos.action_range)[:, 1],
                                                init_std=0.25,       # 第2次学是 0.3
                                                use_orthogonal_init=True),
                        critic=PPOCritic(state_dim=env_pos.state_dim, use_orthogonal_init=True),
                        path=simulationPath)
        agent_att = PPO(env_msg=env_att_msg,
                        actor_lr=5e-5,
                        critic_lr=5e-4,
                        gamma=0.99,
                        K_epochs=25,
                        eps_clip=0.2,
                        action_std_init=action_std_init,
                        buffer_size=buffer_size,
                        max_train_steps=max_train_steps,
                        actor=PPOActor_Gaussian(state_dim=env_att.state_dim,
                                                action_dim=env_att.action_dim,
                                                a_min=np.array(env_att.action_range)[:, 0],
                                                a_max=np.array(env_att.action_range)[:, 1],
                                                init_std=0.15,  # 第2次学是 0.3
                                                use_orthogonal_init=True),
                        critic=PPOCritic(state_dim=env_att.state_dim, use_orthogonal_init=True),
                        path=simulationPath)
        agent_pos.PPO_info()
        agent_att.PPO_info()

        if RETRAIN:
            print('RELOADING......')
            '''如果两次奖励函数不一样，那么必须重新初始化 critic'''
            optPath_pos = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/position_tracking/nets/pos_new1-260/'
            agent_pos.actor.load_state_dict(torch.load(optPath_pos + 'actor'))
            agent_pos.critic.load_state_dict(torch.load(optPath_pos + 'critic'))
            env_pos.load_norm_normalizer_from_file(optPath_pos, 'state_norm.csv')
            # agent_pos.critic.init(True)

            optPath_att = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/attitude_tracking/nets/draw_and_opt/'
            agent_att.actor.load_state_dict(torch.load(optPath_att + 'actor'))
            agent_att.critic.load_state_dict(torch.load(optPath_att + 'critic'))
            env_att.load_norm_normalizer_from_file(optPath_att, 'state_norm.csv')
            # agent_att.critic.init(True)
            '''如果两次奖励函数不一样，那么必须重新初始化 critic'''

        while True:
            '''1. 初始化 buffer 索引和累计奖励记录'''
            r_p = r_a = 0.
            buffer_index = 0
            '''1. 初始化 buffer 索引和累计奖励记录'''

            '''2. 重新开始一次收集数据'''
            while buffer_index < agent_pos.buffer.batch_size:
                if env_pos.is_terminal or env_att.is_terminal:
                    reset_pos_ctrl_param('zero')
                    reset_att_ctrl_param('zero')
                    print('Sumr_pos: %.2f | Sumr_att: %.2f' % (r_p, r_a))
                    r_pl.append(r_p)
                    r_al.append(r_a)
                    r_p = r_a = 0.
                    yyf_A = 1.5 * np.random.choice([-1, 1], 4)
                    yyf_T = 5 * np.ones(4)
                    yyf_phi0 = np.pi / 2 * np.random.choice([-1, 0, 1], 4)
                    yyf = [yyf_A, yyf_T, yyf_phi0]
                    env_pos.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True,
                                                           random_pos0=False,
                                                           new_att_ctrl_param=None,
                                                           new_pos_ctrl_parma=pos_ctrl_param,
                                                           outer_param=None)
                    env_att.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False,
                                                           yaw_fixed=False,
                                                           new_att_ctrl_param=att_ctrl_param,
                                                           outer_param=None)
                else:
                    '''2.1 位置 actor'''
                    env_pos.current_state = env_pos.next_state.copy()
                    a_pos, a_log_prob_pos = agent_pos.choose_action(env_pos.current_state)
                    env_pos.get_param_from_actor(a_pos)

                    '''2.2 姿态 actor'''
                    env_att.current_state = env_att.next_state.copy()
                    env_att.ref[:] = env_pos.att_ref[:]
                    env_att.dot_ref[:] = env_pos.dot_att_ref[:]
                    a_att, a_log_prob_att = agent_att.choose_action(env_att.get_state())
                    env_att.get_param_from_actor(a_att)

                    '''2.3 将 a_att 输入到 env_pos 中'''
                    a_att_cof = np.array([10., 10., 10., 0.1, 0.1, 0.1, 1., 1.])
                    env_pos.set_att_ctrl_from_outer(a_att * a_att_cof)  # 此时 env_pos 的姿态控制器已经更新

                    '''2.4 此时两个 env 的控制参数完全相同了'''
                    action_4_uav = env_pos.generate_action_4_uav()
                    env_pos.step_update(action_4_uav)
                    r_p += env_pos.reward
                    s_p = 1.0 if env_pos.terminal_flag == 1 else 0.0
                    agent_pos.buffer.append(s=env_pos.current_state_norm(env_pos.current_state, update=True),
                                            a=a_pos,
                                            log_prob=a_log_prob_pos,
                                            r=reward_pos_norm(env_pos.reward),
                                            s_=env_pos.next_state_norm(env_pos.next_state, update=True),
                                            done=1.0 if env_pos.is_terminal else 0.0,
                                            success=s_p,
                                            index=buffer_index)

                    torque = env_att.att_control(env_att.ref, env_att.dot_ref, None)
                    env_att.step_update([torque[0], torque[1], torque[2]])
                    r_a += env_att.reward
                    # print('---Reward: %.4f  %.4f' % (env_pos.reward, env_att.reward))
                    s_a = 1.0 if env_att.terminal_flag == 1 else 0.0
                    agent_att.buffer.append(s=env_att.current_state_norm(env_att.current_state, update=True),
                                            a=a_att,
                                            log_prob=a_log_prob_att,
                                            r=reward_att_norm(env_att.reward),
                                            s_=env_att.next_state_norm(env_att.next_state, update=True),
                                            done=1.0 if env_att.is_terminal else 0.0,
                                            success=s_a,
                                            index=buffer_index)
                    buffer_index += 1

                    # env_pos.image = env_pos.image_copy.copy()
                    # env_pos.draw_3d_points_projection(np.atleast_2d([env_pos.uav_pos(), env_pos.pos_ref]), [Color().Red, Color().DarkGreen])
                    # env_pos.draw_time_error(env_pos.uav_pos(), env_pos.pos_ref)
                    # env_pos.show_image(False)
            '''2. 重新开始一次收集数据'''

            '''3. 开始学习'''
            print('~~~~~~~~~~ Training Start~~~~~~~~~~')
            print('Train Epoch: {}'.format(t_epoch))
            timestep += buffer_size
            agent_att.learn(timestep, buf_num=1)
            agent_pos.learn(timestep, buf_num=1)
            agent_att.cnt += 1
            agent_pos.cnt += 1
            '''3. 开始学习'''

            '''4. 每学习 10 次，测试一下'''
            if t_epoch % 10 == 0 and t_epoch > 0:
                n = 10
                print('   Training pause......')
                print('   Testing...')
                test_r = 0.
                for i in range(n):
                    reset_pos_ctrl_param('optimal')
                    reset_att_ctrl_param('optimal')
                    _yyf_A = 1.5 * np.random.choice([-1, 1], 4)
                    _yyf_T = 5 * np.ones(4)
                    _yyf_phi0 = np.pi / 2 * np.random.choice([-1, 0, 1], 4)
                    _yyf = [_yyf_A, _yyf_T, _yyf_phi0]
                    env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True,
                                                            random_pos0=False,
                                                            new_att_ctrl_param=None,
                                                            new_pos_ctrl_parma=pos_ctrl_param,
                                                            outer_param=None)
                    while not env_test.is_terminal:
                        _a_pos = agent_pos.evaluate(env_pos.current_state_norm(env_test.current_state, update=False))

                        _e_att_ = env_test.uav_att() - env_test.att_ref
                        _e_pqr_ = env_test.uav_dot_att() - env_test.dot_att_ref
                        _s = np.concatenate((_e_att_, _e_pqr_))

                        _a_att = agent_att.evaluate(env_att.current_state_norm(_s, update=False))

                        env_test.get_param_from_actor(_a_pos)  # 将控制器参数更新
                        env_test.set_att_ctrl_from_outer(_a_att * a_att_cof)

                        _a_4_uav = env_test.generate_action_4_uav()
                        env_test.step_update(_a_4_uav)
                        test_r += env_test.reward

                        # env_test.image = env_test.image_copy.copy()
                        # env_test.draw_3d_points_projection(np.atleast_2d([env_test.uav_pos(), env_test.pos_ref]), [Color().Red, Color().DarkGreen])
                        # env_test.draw_time_error(env_test.uav_pos(), env_test.pos_ref)
                        # env_test.show_image(False)

                test_num += 1
                test_r /= 10.
                test_reward.append(test_r)
                print('   Evaluating %.0f | Reward: %.2f ' % (test_num, test_r))
                pd.DataFrame({'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
                pd.DataFrame({'sumr_pos': r_pl}).to_csv(simulationPath + 'sumr_pos.csv')
                pd.DataFrame({'sumr_att': r_al}).to_csv(simulationPath + 'sumr_att.csv')
                print('   Testing finished...')
                print('   Go back to training...')
            '''4. 每学习 10 次，测试一下'''

            '''5. 每学习 100 次，减小一次探索概率'''
            if t_epoch % 250 == 0 and t_epoch > 0:
                if agent_pos.actor.std > 0.1:
                    agent_pos.actor.std -= 0.05
                if agent_att.actor.std > 0.1:
                    agent_att.actor.std -= 0.05
            '''5. 每学习 100 次，减小一次探索概率'''

            '''6. 每学习 10 次，保存一下 policy'''
            if t_epoch % 10 == 0 and t_epoch > 0:
                test_num += 1
                print('...check point save...')
                temp = simulationPath + 'trainNum_{}/'.format(t_epoch)
                os.mkdir(temp)
                time.sleep(0.01)
                agent_pos.save_ac(msg='pos_{}'.format(t_epoch), path=temp)
                env_pos.save_state_norm(temp, msg='pos')
                agent_att.save_ac(msg='att_{}'.format(t_epoch), path=temp)
                env_att.save_state_norm(temp, msg='att')
            '''6. 每学习 50 次，保存一下 policy'''

            t_epoch += 1
            # print('~~~~~~~~~~  Training End ~~~~~~~~~~')
    else:
        opt_pos = PPOActor_Gaussian(state_dim=env_test.state_dim,
                                    action_dim=env_test.action_dim)
        optPathPos = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/position_tracking/nets/pos_new1-260/'
        opt_pos.load_state_dict(torch.load(optPathPos + 'actor'))  # 测试时，填入测试actor网络
        agent_pos = PPO(env_msg=env_pos_msg, actor=opt_pos, path=optPathPos)
        env_test.load_norm_normalizer_from_file(optPathPos, 'state_norm.csv')

        opt_att = PPOActor_Gaussian(state_dim=env_att.state_dim,
                                    action_dim=env_att.action_dim)
        optPathAtt = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/attitude_tracking/nets/draw_and_opt/'
        opt_att.load_state_dict(torch.load(optPathAtt + 'actor'))  # 测试时，填入测试actor网络
        agent_att = PPO(env_msg=env_att_msg, actor=opt_att, path=optPathAtt)
        att_norm = get_normalizer_from_file(env_att.state_dim, optPathAtt, 'state_norm.csv')
        # env_att.load_norm_normalizer_from_file(optPathAtt, 'state_norm.csv')

        n = 10
        SMC = False
        for i in range(n):
            opt_SMC_para_pos = np.atleast_2d(np.zeros(env_pos.action_dim))
            opt_SMC_para_att = np.atleast_2d(np.zeros(env_att.action_dim))
            env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True,
                                                    random_pos0=False,
                                                    new_att_ctrl_param=None,
                                                    new_pos_ctrl_parma=pos_ctrl_param,
                                                    outer_param=None)
            env_test.show_image(False)
            test_r = 0.
            while not env_test.is_terminal:
                _a_pos = agent_pos.evaluate(env_test.current_state_norm(env_test.current_state, update=False))
                _e_att = env_test.uav_att() - env_test.att_ref
                _e_dot_att = env_test.uav_dot_att() - env_test.dot_att_ref
                _s_4_att = np.concatenate((_e_att, _e_dot_att))
                _a_att = agent_att.evaluate(att_norm(_s_4_att, update=False))

                if not SMC:
                    env_test.get_param_from_actor(_a_pos)  # update params of POSITION controller
                    env_test.set_att_ctrl_from_outer(_a_att * a_att_cof)    # update params of ATTITUDE controller

                opt_SMC_para_pos = np.insert(opt_SMC_para_pos, opt_SMC_para_pos.shape[0], _a_pos, axis=0)
                opt_SMC_para_att = np.insert(opt_SMC_para_att, opt_SMC_para_att.shape[0], _a_att * a_att_cof, axis=0)

                _a_4_uav = env_test.generate_action_4_uav()
                env_test.step_update(_a_4_uav)
                test_r += env_test.reward

                env_test.image = env_test.image_copy.copy()
                env_test.draw_3d_points_projection(np.atleast_2d([env_test.uav_pos(), env_test.pos_ref]), [Color().Red, Color().DarkGreen])
                env_test.draw_time_error(env_test.uav_pos(), env_test.pos_ref)
                env_test.show_image(False)
            print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))

            plot_SMC_param(opt_SMC_para_pos, 'pos')
            plot_SMC_param(opt_SMC_para_att,'att')
            plt.show()
