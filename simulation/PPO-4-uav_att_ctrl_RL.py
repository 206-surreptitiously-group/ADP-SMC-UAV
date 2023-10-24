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
ENV = 'uav_att_ctrl_RL'
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
uav_param.pos_zone = np.atleast_2d([[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-90), deg2rad(90)], [deg2rad(-90), deg2rad(90)], [deg2rad(-180), deg2rad(180)]])
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
    log_dir = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
    os.mkdir(simulationPath)
    c = cv.waitKey(1)
    TRAIN = True  # 直接训练
    RETRAIN = False  # 基于之前的训练结果重新训练
    TEST = not TRAIN

    env = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    reset_att_ctrl_param('zero')
    env.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False,  yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)

    env_test = uav_att_ctrl_RL(uav_param, att_ctrl_param)
    reset_att_ctrl_param('zero')
    env_test.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False,  yaw_fixed=False, new_att_ctrl_param=att_ctrl_param)

    reward_norm = Normalization(shape=1)

    if TRAIN:
        action_std_init = 0.4  # 初始探索方差
        min_action_std = 0.05  # 最小探索方差
        max_train_steps = int(5e6)
        buffer_size = int(env.time_max / env.dt) * 2
        K_epochs = 50
        timestep = 0
        t_epoch = 0  # 当前训练次数
        test_num = 0

        agent = PPO(env=env,
                    actor_lr=1e-4,
                    critic_lr=1e-3,
                    gamma=0.99,
                    K_epochs=K_epochs,
                    eps_clip=0.2,
                    action_std_init=action_std_init,
                    buffer_size=buffer_size,
                    max_train_steps=max_train_steps,
                    actor=PPOActor_Gaussian(state_dim=env.state_dim,
                                            action_dim=env.action_dim,
                                            a_min=np.array(env.action_range)[:, 0],
                                            a_max=np.array(env.action_range)[:, 1],
                                            init_std=0.7,  # 第2次学是 0.3
                                            use_orthogonal_init=True),
                    critic=PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True),
                    path=simulationPath)
        agent.PPO_info()

        if RETRAIN:
            print('RELOADING......')
            '''如果两次奖励函数不一样，那么必须重新初始化 critic'''
            optPath = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/temp/'
            agent.actor.load_state_dict(torch.load(optPath + 'actor_trainNum_300'))  # 测试时，填入测试actor网络
            # agent.critic.load_state_dict(torch.load(optPath + 'critic_trainNum_300'))
            agent.critic.init(True)
            '''如果两次奖励函数不一样，那么必须重新初始化 critic'''

        while True:
            '''1. 初始化 buffer 索引和累计奖励记录'''
            sumr = 0.
            buffer_index = 0
            '''1. 初始化 buffer 索引和累计奖励记录'''

            '''2. 收集数据'''
            while buffer_index < agent.buffer.batch_size:
                if env.is_terminal:  # 如果某一个回合结束
                    reset_att_ctrl_param('zero')
                    # if t_epoch % 10 == 0 and t_epoch > 0:
                    print('Sumr:  ', sumr)
                    sumr_list.append(sumr)
                    sumr = 0.
                    env.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False,
                                                       yaw_fixed=False,
                                                       new_att_ctrl_param=att_ctrl_param)
                else:
                    env.current_state = env.next_state.copy()  # 此时相当于时间已经来到了下一拍，所以 current 和 next 都得更新
                    a, a_log_prob = agent.choose_action(env.current_state)
                    # new_SMC_param = agent.action_linear_trans(a)	# a 肯定在 [-1, 1]
                    new_SMC_param = a.copy()
                    env.get_param_from_actor(new_SMC_param)

                    rhod, dot_rhod, dot2_rhod, _ = ref_inner(env.time,
                                                             env.ref_att_amplitude,
                                                             env.ref_att_period,
                                                             env.ref_att_bias_a,
                                                             env.ref_att_bias_phase)
                    torque = env.att_control(rhod, dot_rhod, None)
                    env.step_update([torque[0], torque[1], torque[2]])
                    sumr += env.reward
                    success = 1.0 if env.terminal_flag == 1 else 0.0
                    agent.buffer.append(s=env.current_state_norm(env.current_state, update=True),
                                        a=a,  # a
                                        log_prob=a_log_prob,  # a_lp
                                        # r=env.reward,							# r
                                        r=reward_norm(env.reward),  # 这里使用了奖励归一化
                                        s_=env.next_state_norm(env.next_state, update=True),
                                        done=1.0 if env.is_terminal else 0.0,  # done
                                        success=success,  # 固定时间内，不出界，就是 success
                                        index=buffer_index  # index
                                        )
                    buffer_index += 1
            '''2. 收集数据'''

            '''3. 开始学习'''
            print('~~~~~~~~~~ Training Start~~~~~~~~~~')
            print('Train Epoch: {}'.format(t_epoch))
            timestep += buffer_size
            agent.learn(timestep, buf_num=1)    # 使用 RolloutBuffer2
            agent.cnt += 1

            '''4. 每学习 10 次，测试一下'''
            if t_epoch % 10 == 0 and t_epoch > 0:
                n = 5
                print('   Training pause......')
                print('   Testing...')
                for i in range(n):
                    reset_att_ctrl_param('zero')
                    env_test.reset_uav_att_ctrl_RL_tracking(random_trajectroy=False,
                                                            yaw_fixed=False,
                                                            new_att_ctrl_param=att_ctrl_param)
                    test_r = 0.
                    while not env_test.is_terminal:
                        _a = agent.evaluate(env.current_state_norm(env_test.current_state, update=False))
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
                    test_num += 1
                    test_reward.append(test_r)
                    print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
                pd.DataFrame({'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
                pd.DataFrame({'sumr_list': sumr_list}).to_csv(simulationPath + 'sumr_list.csv')
                print('   Testing finished...')
                print('   Go back to training...')
            '''4. 每学习 10 次，测试一下'''

            '''5. 每学习 100 次，减小一次探索概率'''
            if t_epoch % 250 == 0 and t_epoch > 0:
                if agent.actor.std > 0.1:
                    agent.actor.std -= 0.05
            '''5. 每学习 100 次，减小一次探索概率'''

            '''6. 每学习 50 次，保存一下 policy'''
            if t_epoch % 50 == 0 and t_epoch > 0:
                # 	average_test_r = agent.agent_evaluate(5)
                test_num += 1
                print('...check point save...')
                temp = simulationPath + 'trainNum_{}/'.format(t_epoch)
                os.mkdir(temp)
                time.sleep(0.01)
                agent.save_ac(msg='trainNum_{}'.format(t_epoch), path=temp)
                env.save_state_norm(temp)
            '''6. 每学习 50 次，保存一下 policy'''

            t_epoch += 1
            print('~~~~~~~~~~  Training End ~~~~~~~~~~')
    else:
        pass
