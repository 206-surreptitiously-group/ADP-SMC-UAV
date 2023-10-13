import os
import sys
import datetime
import time
import cv2 as cv
import pandas as pd
import torch


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from environment.envs.UAV.uav_pos_ctrl_RL import uav_pos_ctrl_RL, uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from environment.Color import Color
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from common.common_cls import *
from common.common_func import *

optPath = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/nets/'
show_per = 1
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


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


# setup_seed(3407)


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


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
		super(PPOActorCritic, self).__init__()
		self.checkpoint_file = chkpt_dir + name + '_ppo'
		self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
		self.action_dim = _action_dim
		# 应该是初始化方差，一个动作就一个方差，两个动作就两个方差，std 是标准差
		self.action_var = torch.full((_action_dim,), _action_std_init * _action_std_init)
		self.actor = nn.Sequential(
			nn.Linear(_state_dim, 256),
			nn.Tanh(),
			nn.Linear(256, 128),
			nn.Tanh(),
			nn.Linear(128, 64),
			nn.Tanh(),
			nn.Linear(64, _action_dim),
			nn.ReLU()  # 因为是参数优化，所以最后一层用ReLU
		)
		# nn.init.orthogonal(self.actor)
		self.critic = nn.Sequential(
			nn.Linear(_state_dim, 256),
			nn.Tanh(),
			nn.Linear(256, 128),
			nn.Tanh(),
			nn.Linear(128, 64),
			nn.Tanh(),
			nn.Linear(64, 1)
		)
		self.actor_reset_orthogonal()
		self.critic_reset_orthogonal()

		# self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = 'cpu'
		# torch.cuda.empty_cache()
		self.to(self.device)

	def actor_reset_orthogonal(self):
		nn.init.orthogonal_(self.actor[0].weight, gain=1.0)
		nn.init.constant_(self.actor[0].bias, 1e-3)
		nn.init.orthogonal_(self.actor[2].weight, gain=1.0)
		nn.init.constant_(self.actor[2].bias, 1e-3)
		nn.init.orthogonal_(self.actor[4].weight, gain=0.01)
		nn.init.constant_(self.actor[4].bias, 1e-3)
		nn.init.orthogonal_(self.actor[6].weight, gain=0.01)
		nn.init.constant_(self.actor[6].bias, 1e-3)

	def critic_reset_orthogonal(self):
		nn.init.orthogonal_(self.critic[0].weight, gain=1.0)
		nn.init.constant_(self.critic[0].bias, 1e-3)
		nn.init.orthogonal_(self.critic[2].weight, gain=1.0)
		nn.init.constant_(self.critic[2].bias, 1e-3)
		nn.init.orthogonal_(self.critic[4].weight, gain=1.0)
		nn.init.constant_(self.critic[4].bias, 1e-3)
		nn.init.orthogonal_(self.critic[6].weight, gain=1.0)
		nn.init.constant_(self.critic[6].bias, 1e-3)

	def set_action_std(self, new_action_std):
		self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

	def forward(self):
		raise NotImplementedError

	def act(self, _s):
		action_mean = self.actor(_s)  # PPO 给出的是分布，所以这里直接拿出的只能是 mean
		cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # 协方差矩阵
		dist = MultivariateNormal(action_mean, cov_mat)

		_a = dist.sample()
		# _a = torch.clip(_a, 0, torch.inf)
		action_logprob = dist.log_prob(_a)
		state_val = self.critic(_s)

		return _a.detach(), action_logprob.detach(), state_val.detach()

	def evaluate(self, _s, a):
		action_mean = self.actor(_s)
		action_var = self.action_var.expand_as(action_mean)
		cov_mat = torch.diag_embed(action_var).to(self.device)
		dist = MultivariateNormal(action_mean, cov_mat)

		# For Single Action Environments.
		if self.action_dim == 1:
			a = a.reshape(-1, self.action_dim)

		action_logprobs = dist.log_prob(a)
		dist_entropy = dist.entropy()
		state_values = self.critic(_s)

		return action_logprobs, state_values, dist_entropy

	def save_checkpoint(self, name=None, path='', num=None):
		print('...saving checkpoint...')
		if name is None:
			torch.save(self.state_dict(), self.checkpoint_file)
		else:
			if num is None:
				torch.save(self.state_dict(), path + name)
			else:
				torch.save(self.state_dict(), path + name + str(num))

	def save_all_net(self):
		print('...saving all net...')
		torch.save(self, self.checkpoint_file_whole_net)

	def load_checkpoint(self):
		print('...loading checkpoint...')
		self.load_state_dict(torch.load(self.checkpoint_file))


if __name__ == '__main__':
	log_dir = os.path.dirname(os.path.abspath(__file__)) + '/../datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)
	TRAIN = False  # 直接训练
	RETRAIN = False  # 基于之前的训练结果重新训练
	TEST = not TRAIN

	env = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
	reset_pos_ctrl_param('zero')
	env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)
	env.show_image(True)

	env_test = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)
	reset_pos_ctrl_param('optimal')
	env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False, random_pos0=False, new_att_ctrl_param=None, new_pos_ctrl_parma=pos_ctrl_param)

	if TRAIN:
		action_std_init = 0.6	# 初始探索方差
		min_action_std = 0.1	# 最小探索方差
		max_t_epoch = 1e4		# 最大训练次数
		t_epoch = 0				# 当前训练次数
		action_std_decay_freq = int(500)	# 每训练 500 次，减小一次探索方差
		action_std_decay_rate = 0.05  # 每次减小方差的数值
		test_num = 0

		'''重新加载Policy网络结构，这是必须的操作'''
		policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulationPath)
		policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulationPath)
		# optimizer = torch.optim.Adam([
		# 	{'params': agent.policy.actor.parameters(), 'lr': agent.actor_lr},
		# 	{'params': agent.policy.critic.parameters(), 'lr': agent.critic_lr}
		# ])
		'''重新加载Policy网络结构，这是必须的操作'''

		agent = PPO(env=env,
					actor_lr=1e-4,
					critic_lr=1e-3,
					gamma=0.99,
					K_epochs=8,
					eps_clip=0.2,
					action_std_init=action_std_init,
					buffer_size=1024,  # 假设可以包含两条完整的最长时间的轨迹
					policy=policy,
					policy_old=policy_old,
					path=simulationPath)
		agent.PPO_info()

		while t_epoch < max_t_epoch:
			'''1. 初始化 buffer 索引和累计奖励记录'''
			sumr = 0.
			buffer_index = 0

			'''2. 重新开始一次收集数据'''
			while buffer_index < agent.buffer.batch_size:
				if env.is_terminal:		# 如果某一个回合结束
					reset_pos_ctrl_param('zero')
					env.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False,
													   random_pos0=False,
													   new_att_ctrl_param=None,
													   new_pos_ctrl_parma=pos_ctrl_param)
				else:
					env.current_state = env.next_state.copy()
					action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)  # 待探索，无梯度
					env.get_param_from_actor(action_from_actor.detach().cpu().numpy().flatten())  # 将控制器参数更新
					action_4_uav = env.generate_action_4_uav()	# 生成无人机物理控制量
					env.step_update(action_4_uav)  # 环境更新的 action 需要是物理的 action
					sumr += env.reward

					agent.buffer.append(s=env.current_state,
										a=action_from_actor,  # .cpu().numpy()
										log_prob=a_log_prob.numpy(),
										r=env.reward,
										sv=s_value.numpy(),
										done=1.0 if env.is_terminal else 0.0,
										index=buffer_index)
					buffer_index += 1

			'''3. 开始一次新的学习'''
			print('~~~~~~~~~~ Training Start~~~~~~~~~~')
			print('Train Epoch: {}'.format(t_epoch))
			agent.learn()

			'''每学习 10 次，测试一下'''
			if t_epoch % 10 == 0 and t_epoch > 0:
				n = 1
				print('   Training pause......')
				print('   Testing...')
				for i in range(n):
					reset_pos_ctrl_param('zero')
					env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=False,
															random_pos0=False,
															new_att_ctrl_param=None,
															new_pos_ctrl_parma=pos_ctrl_param)
					test_r = 0.
					# print('init state', env_test.uav_state_call_back())
					while not env_test.is_terminal:
						_a = agent.evaluate(env_test.current_state)
						env_test.get_param_from_actor(_a.detach().cpu().numpy().flatten())  # 将控制器参数更新
						_a_4_uav = env_test.generate_action_4_uav()
						env_test.step_update(_a_4_uav)
						test_r += env_test.reward
						env_test.image = env_test.image_copy.copy()
						env_test.draw_3d_points_projection(np.atleast_2d([env_test.uav_pos(), env_test.pos_ref]), [Color().Red, Color().DarkGreen])
						env_test.draw_time_error(env_test.uav_pos(), env_test.pos_ref)
						env_test.show_image(False)
					test_num += 1
					test_reward.append(test_r)
					print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
				pd.DataFrame({'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
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
				agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=t_epoch)
			'''每学习 50 次，保存一下 policy'''

			'''每学习 500 次，减小一次探索方差'''
			if t_epoch % action_std_decay_freq == 0 and t_epoch > 0:
				print('......action_std_decay......')
				agent.decay_action_std(action_std_decay_rate, min_action_std)
			'''每学习 500 次，减小一次探索方差'''

			t_epoch += 1
			print('Sumr: %.2f' % (sumr))
			print('~~~~~~~~~~  Training End ~~~~~~~~~~')
	else:
		policy = PPOActorCritic(env.state_dim, env.action_dim, 0.6, 'Policy', simulationPath)
		policy_old = PPOActorCritic(env.state_dim, env.action_dim, 0.6, 'Policy_old', simulationPath)
		agent = PPO(env=env_test, policy=policy, policy_old=policy_old, path=simulationPath)
		agent.policy.load_state_dict(torch.load(optPath + 'Policy_PPO2250'))
		n = 100
		for i in range(n):
			reset_pos_ctrl_param('optimal')
			env_test.reset_uav_pos_ctrl_RL_tracking(random_trajectroy=True,
													random_pos0=True,
													new_att_ctrl_param=None,
													new_pos_ctrl_parma=pos_ctrl_param)
			env_test.show_image(False)
			test_r = 0.
			optimal_param = np.zeros(8)
			while not env_test.is_terminal:
				_a = agent.evaluate(env_test.current_state)
				env_test.get_param_from_actor(_a.detach().cpu().numpy().flatten())  # 将控制器参数更新
				_a_4_uav = env_test.generate_action_4_uav()
				env_test.step_update(_a_4_uav)
				test_r += env_test.reward

				# env_test.image = env_test.image_copy.copy()
				# env_test.draw_3d_points_projection(np.atleast_2d([env_test.uav_pos(), env_test.pos_ref]), [Color().Red, Color().DarkGreen])
				# env_test.draw_time_error(env_test.uav_pos(), env_test.pos_ref)
				# env_test.show_image(False)
			print('   Evaluating %.0f | Reward: %.2f ' % (i, test_r))
