import os
import sys
import datetime
import time
import cv2 as cv
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.envs.FlightAttitudeSimulator.flight_attitude_simulator_continuous import Flight_Attitude_Simulator_Continuous as FAS
from environment.Color import Color
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from algorithm.policy_base.Proximal_Policy_Optimization import PPOActor_Gaussian, PPOCritic
from common.common_cls import *
from common.common_func import *

optPath = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/train1/'
show_per = 1
timestep = 0
ENV = 'FAS_continuous'
ALGORITHM = 'PPO'

test_episode = []
sumr_list = []
test_reward = []


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

    env = FAS(-60, 0.)

    reward_norm = Normalization(shape=1)
    reward_scale = RewardScaling(shape=1, gamma=0.99)

    if TRAIN:
        action_std_init = 0.4	# 初始探索方差
        min_action_std = 0.05	# 最小探索方差
        max_train_steps = int(5e6)
        buffer_size = int(env.timeMax / env.dt) * 2
        K_epochs = 50
        timestep = 0
        t_epoch = 0				# 当前训练次数
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
                    actor=PPOActor_Gaussian(state_dim=env.state_dim, action_dim=env.action_dim, use_orthogonal_init=True),
                    critic=PPOCritic(state_dim=env.state_dim, use_orthogonal_init=True),
                    path=simulationPath)
        agent.PPO_info()

        # while t_epoch < max_t_epoch:
        while True:
            '''1. 初始化 buffer 索引和累计奖励记录'''
            sumr = 0.
            buffer_index = 0

            '''2. 重新开始一次收集数据'''
            while buffer_index < agent.buffer.batch_size:
                if env.is_terminal:		# 如果某一个回合结束
                    sumr_list.append(sumr)
                    # print('Sumr: %.2f' % sumr)
                    sumr = 0.
                    env.reset_random()
                else:
                    env.current_state = env.next_state.copy()				# 此时相当于时间已经来到了下一拍，所以 current 和 next 都得更新
                    action_from_actor, a_log_prob = agent.choose_action(env.current_state)  # 加了探索，返回的数都是 .detach().cpu().numpy().flatten() 之后的
                    a = agent.action_linear_trans(action_from_actor)
                    env.step_update(a)  # 环境更新的 action 需要是物理的 action
                    sumr += env.reward
                    # if env.reward > 0:
                    #     print('日乐购')
                    success = 1.0 if env.terminal_flag == 4 else 0.0
                    agent.buffer.append(s=env.current_state,					# s
                                        a=action_from_actor,					# a
                                        log_prob=a_log_prob,					# a_lp
                                        # r=env.reward,							# r
                                        r=reward_norm(env.reward),				# 这里使用了奖励归一化
                                        s_=env.next_state,						# s'
                                        done=1.0 if env.is_terminal else 0.0,	# done
                                        success=0.,                             # success 轨迹跟踪，没有 success 的概念
                                        index=buffer_index						# index
                                        )

                    buffer_index += 1

            '''3. 开始一次新的学习'''
            print('~~~~~~~~~~ Training Start~~~~~~~~~~')
            print('Train Epoch: {}'.format(t_epoch))
            timestep += buffer_size
            agent.learn(timestep)

            '''每学习 10 次，测试一下'''
            if t_epoch % 10 == 0 and t_epoch > 0:
                n = 5
                print('   Training pause......')
                print('   Testing...')
                for i in range(n):
                    env.reset_random()
                    test_r = 0.
                    # print('init state', env_test.uav_state_call_back())
                    while not env.is_terminal:
                        action_from_actor = agent.evaluate(env.current_state)
                        _a = agent.action_linear_trans(action_from_actor)
                        env.step_update(_a)
                        env.show_dynamic_image(isWait=False)
                        test_r += env.reward
                    test_num += 1
                    test_reward.append(test_r)
                    print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
                pd.DataFrame({'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
                pd.DataFrame({'reward': sumr_list}).to_csv(simulationPath + 'sumr_list.csv')
                print('   Testing finished...')
                print('   Go back to training...')
            '''每学习 10 次，测试一下'''

            '''每学习 50 次，保存一下 policy'''
            if t_epoch % 50 == 0 and t_epoch > 0:
                # 	average_test_r = agent.agent_evaluate(5)
                test_num += 1
                print('...check point save...')
                temp = simulationPath + 'trainNum_{}_episode_{}/'.format(t_epoch, agent.episode)
                os.mkdir(temp)
                time.sleep(0.01)
                agent.save_ac(msg='trainNum_{}_episode_{}'.format(t_epoch, agent.episode), path=temp)
            '''每学习 50 次，保存一下 policy'''

            t_epoch += 1
            print('~~~~~~~~~~  Training End ~~~~~~~~~~')
