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
from algorithm.actor_critic.Twin_Delayed_DDPG import Twin_Delayed_DDPG as TD3
from common.common_cls import *
from common.common_func import *

optPath = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/train2/'
show_per = 1
timestep = 0
ENV = 'uav_pos_ctrl_RL'
ALGORITHM = 'TD3'

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
    else:	# optimal
        pos_ctrl_param.k1 = np.array([1.2, 0.8, 0.5])
        pos_ctrl_param.k2 = np.array([0.2, 0.6, 0.5])
        pos_ctrl_param.gamma = np.array([0.2, 0.2, 0.2])
        pos_ctrl_param.lmd = np.array([2.0, 2.0, 2.0])


class Critic(nn.Module):
    def __init__(self, beta, state_dim, action_dim, name, chkpt_dir):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 64)  # state -> hidden1

        self.fc2 = nn.Linear(64, 64)  # hidden1 -> hidden2

        self.action_value = nn.Linear(self.action_dim, 64)  # action -> hidden2
        self.q = nn.Linear(64, 1)  # hidden2 -> output action value

        self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, _action):
        state_value = self.fc1(state)  # forward
        state_value = func.relu(state_value)  # relu
        state_value = self.fc2(state_value)

        # action_value = func.relu(self.action_value(_action))      # 原来的

        action_value = self.action_value(_action)

        state_action_value = func.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)


class Actor(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, name, chkpt_dir):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.checkpoint_file = chkpt_dir + name + '_ddpg'
        self.checkpoint_file_whole_net = chkpt_dir + name + '_ddpgALL'

        self.fc1 = nn.Linear(self.state_dim, 128)  # 输入 -> 第一个隐藏层
        self.fc2 = nn.Linear(128, 64)  # 第一个隐藏层 -> 第二个隐藏层
        self.mu = nn.Linear(64, self.action_dim)  # 第二个隐藏层 -> 输出层

        self.initialization()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def initialization(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        f3 = 0.003
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

    def forward(self, state):
        x = func.relu(self.fc1(state))
        x = func.relu(self.fc2(x))
        x = torch.tanh(self.mu(x))  # bound the output to [-1, 1]
        return x


def fullFillReplayMemory_with_Optimal(randomEnv: bool,
                                      fullFillRatio: float,
                                      is_only_success: bool):
    print('Retraining...')
    print('Collecting...')
    agent.load_models(optPath)
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    _new_state, _new_action, _new_reward, _new_state_, _new_done = [], [], [], [], []
    while agent.memory.mem_counter < fullFillCount:
        env.reset_random() if randomEnv else env.reset()
        _new_state.clear()
        _new_action.clear()
        _new_reward.clear()
        _new_state_.clear()
        _new_done.clear()
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = agent.choose_action(env.current_state, is_optimal=False)
            _action = agent.action_linear_trans(_action_from_actor)
            env.step_update(_action)
            # env.show_dynamic_image(isWait=False)
            if is_only_success:
                _new_state.append(env.current_state)
                _new_action.append(env.current_action)
                _new_reward.append(env.reward)
                _new_state_.append(env.next_state)
                _new_done.append(1.0 if env.is_terminal else 0.0)
            else:
                agent.memory.store_transition(np.array(env.current_state),
                                              np.array(env.current_action),
                                              np.array(env.reward),
                                              np.array(env.next_state),
                                              1 if env.is_terminal else 0)
        if is_only_success:
            if env.terminal_flag == 3:
                print('Update Replay Memory......')
                agent.memory.store_transition_per_episode(_new_state, _new_action, _new_reward, _new_state_, _new_done)


def fullFillReplayMemory_Random(randomEnv: bool, fullFillRatio: float):
    """
    :param randomEnv:       init env randomly
    :param fullFillRatio:   the ratio
    :return:                None
    """
    print('Collecting...')
    fullFillCount = int(fullFillRatio * agent.memory.mem_size)
    fullFillCount = max(min(fullFillCount, agent.memory.mem_size), agent.memory.batch_size)
    while agent.memory.mem_counter < fullFillCount:
        if randomEnv:
            reset_pos_ctrl_param('zero')
            env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False,
                                               random_pos0=True,
                                               new_att_ctrl_param=None,
                                               new_pos_ctrl_parma=pos_ctrl_param)
        else:
            reset_pos_ctrl_param('zero')
            env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False,
                                               random_pos0=False,
                                               new_att_ctrl_param=None,
                                               new_pos_ctrl_parma=pos_ctrl_param)
        while not env.is_terminal:
            if agent.memory.mem_counter % 100 == 0:
                print('replay_count = ', agent.memory.mem_counter)
            env.current_state = env.next_state.copy()  # 状态更新
            _action_from_actor = agent.choose_action(env.current_state, False)
            _new_SMC_param = agent.action_linear_trans(_action_from_actor)
            env.get_param_from_actor(_new_SMC_param)
            _action_4_uav = env.generate_action_4_uav()
            env.step_update(_action_4_uav)
            agent.memory.store_transition(np.array(env.current_state),
                                          _action_from_actor,
                                          np.array(env.reward),
                                          np.array(env.next_state),
                                          1 if env.is_terminal else 0)


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

    env = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
    reset_pos_ctrl_param('zero')
    env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False, random_pos0=True, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)
    env.show_image(True)

    env_test = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
    reset_pos_ctrl_param('zero')
    env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False, random_pos0=True, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)

    if TRAIN:
        actor = Actor(1e-4, env.state_dim, env.action_dim, 'Actor', simulationPath)
        target_actor = Actor(1e-4, env.state_dim, env.action_dim, 'TargetActor', simulationPath)
        critic1 = Critic(1e-3, env.state_dim, env.action_dim, 'Critic1', simulationPath)
        target_critic1 = Critic(1e-3, env.state_dim, env.action_dim, 'TargetCritic1', simulationPath)
        critic2 = Critic(1e-3, env.state_dim, env.action_dim, 'Critic2', simulationPath)
        target_critic2 = Critic(1e-3, env.state_dim, env.action_dim, 'TargetCritic2', simulationPath)

        agent = TD3(env=env,
                    gamma=0.99,
                    noise_clip=1 / 2, noise_policy=1 / 4, policy_delay=3,
                    critic1_soft_update=1e-2,
                    critic2_soft_update=1e-2,
                    actor_soft_update=1e-2,
                    memory_capacity=40000,  # 20000
                    batch_size=1024,  # 1024
                    use_grad_clip=True,
                    actor=actor,
                    target_actor=target_actor,
                    critic1=critic1,
                    target_critic1=target_critic1,
                    critic2=critic2,
                    target_critic2=target_critic2,
                    path=simulationPath)
        agent.TD3_info()

        if RETRAIN:
            print('Retraining')
            fullFillReplayMemory_with_Optimal(randomEnv=True,
                                              fullFillRatio=0.5,
                                              is_only_success=False)
            # 如果注释掉，就是在上次的基础之上继续学习，如果不是就是重新学习，但是如果两次的奖励函数有变化，那么就必须执行这两句话
            '''生成初始数据之后要再次初始化网络'''
            agent.actor.initialization()
            agent.target_actor.initialization()
            agent.critic1.initialization()
            agent.target_critic1.initialization()
            agent.critic2.initialization()
            agent.target_critic2.initialization()
            '''生成初始数据之后要再次初始化网络'''
        else:
            '''fullFillReplayMemory_Random'''
            fullFillReplayMemory_Random(randomEnv=True, fullFillRatio=0.1)
            '''fullFillReplayMemory_Random'''

        print('Start to train...')
        new_state, new_action, new_reward, new_state_, new_done = [], [], [], [], []
        step = 0
        is_storage_only_success = True
        while True:
            '''回合初始化'''
            reset_pos_ctrl_param('zero')
            sumr = 0.
            env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False,
                                               random_pos0=True,
                                               new_att_ctrl_param=None,
                                               new_pos_ctrl_parma=pos_ctrl_param)
            '''回合初始化'''
            new_state.clear()
            new_action.clear()
            new_reward.clear()
            new_state_.clear()
            new_done.clear()

            while not env.is_terminal:
                c = cv.waitKey(1)
                env.current_state = env.next_state.copy()
                if random.uniform(0, 1) < 0.2:  # 有一定探索概率完全随机探索
                    # print('...random...')
                    action_from_actor = agent.choose_action_random()  # 有一定探索概率完全随机探索
                else:
                    action_from_actor = agent.choose_action(env.current_state, False, sigma=1 / 3)  # 剩下的是神经网络加噪声
                new_SMC_param = agent.action_linear_trans(action_from_actor)  # 将动作转换到实际范围上
                env.get_param_from_actor(new_SMC_param)  # 将控制器参数更新
                action_4_uav = env.generate_action_4_uav()  # 生成无人机物理控制量
                env.step_update(action_4_uav)  # 环境更新的 action 需要是物理的 action
                agent.saveData_Step_Reward(step=step, reward=env.reward, is2file=False, filename='StepReward.csv')
                step += 1
                sumr += env.reward

                if is_storage_only_success:
                    new_state.append(env.current_state)
                    new_action.append(action_from_actor)
                    new_reward.append(env.reward)
                    new_state_.append(env.next_state)
                    new_done.append(1.0 if env.is_terminal else 0.0)
                else:
                    agent.memory.store_transition(np.array(env.current_state),
                                                  np.array(action_from_actor),
                                                  np.array(env.reward),
                                                  np.array(env.next_state),
                                                  1 if env.is_terminal else 0)
                agent.learn(is_reward_ascent=False)

            '''跳出循环代表回合结束'''
            if is_storage_only_success:
                if env.terminal_flag == 1:      # 超时，说明不会太离谱，能把10秒钟跑完
                    print('Update Replay Memory......')
                    for _ in range(5):
                        agent.memory.store_transition_per_episode(new_state, new_action, new_reward, new_state_, new_done)
            '''跳出循环代表回合结束'''

            print(
                '=========START=========',
                'Episode:', agent.episode,
                'Cumulative reward:', round(sumr, 3),
                '==========END=========\n')
            agent.episode += 1

            if agent.episode % 10 == 0:
                temp = simulationPath + 'episode_{}/'.format(agent.episode)
                os.mkdir(temp)
                time.sleep(0.01)
                agent.save_nets(msg='episode_{}'.format(agent.episode), path=temp)

                n = 5
                print('   Training pause......')
                print('   Testing...')
                for i in range(n):
                    reset_pos_ctrl_param('zero')
                    env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False,
                                                            random_pos0=True,
                                                            new_att_ctrl_param=None,
                                                            new_pos_ctrl_parma=pos_ctrl_param)
                    test_r = 0.
                    while not env_test.is_terminal:
                        _a = agent.evaluate(env_test.current_state)
                        _new_SMC_param = agent.action_linear_trans(_a)
                        env_test.get_param_from_actor(_new_SMC_param)  # 将控制器参数更新
                        _a_4_uav = env_test.generate_action_4_uav()
                        env_test.step_update(_a_4_uav)
                        test_r += env_test.reward
                        env_test.image = env_test.image_copy.copy()
                        env_test.draw_3d_points_projection(np.atleast_2d([env_test.uav_pos(), env_test.pos_ref]), [Color().Red, Color().DarkGreen])
                        env_test.draw_time_error(env_test.uav_pos(), env_test.pos_ref)
                        env_test.show_image(False)
                    test_reward.append(test_r)
                    print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
                pd.DataFrame({'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
                print('   Testing finished...')
                print('   Go back to training...')
    else:
        pass
