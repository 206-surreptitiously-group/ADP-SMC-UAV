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
# from environment.envs.UAV.ref_cmd import *
from environment.Color import Color
from algorithm.policy_base.Proximal_Policy_Optimization import Proximal_Policy_Optimization as PPO
from common.common_cls import *
from common.common_func import *

optPath = '../../../datasave/network/'
show_per = 1
timestep = 0
ENV = 'uav_pos_ctrl_RL'
ALGORITHM = 'PPO'

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
uav_param.time_max = 30
uav_param.pos_zone = np.atleast_2d([[-3, 3], [-3, 3], [0, 3]])
uav_param.att_zone = np.atleast_2d([[deg2rad(-45), deg2rad(45)], [deg2rad(-45), deg2rad(45)], [deg2rad(-120), deg2rad(120)]])
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


class PPOActorCritic(nn.Module):
	def __init__(self, _state_dim, _action_dim, _action_std_init, name='PPOActorCritic', chkpt_dir=''):
		super(PPOActorCritic, self).__init__()
		self.checkpoint_file = chkpt_dir + name + '_ppo'
		self.checkpoint_file_whole_net = chkpt_dir + name + '_ppoALL'
		self.action_dim = _action_dim
		# 应该是初始化方差，一个动作就一个方差，两个动作就两个方差，std 是标准差
		self.action_var = torch.full((_action_dim,), _action_std_init * _action_std_init)
		self.actor = nn.Sequential(
			nn.Linear(_state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, _action_dim),
			nn.ReLU()  # 因为是参数优化，所以最后一层用ReLU
		)
		self.critic = nn.Sequential(
			nn.Linear(_state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, 1)
		)
		# self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		self.device = 'cpu'
		# torch.cuda.empty_cache()
		self.to(self.device)

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
	TRAIN = True  # 直接训练
	RETRAIN = False  # 基于之前的训练结果重新训练
	TEST = not TRAIN

	'''随机初始化位置控制参数: 3 个 k1, 3 个 k2, 1 个 gamma, 1 个 lambda'''
	ALL_ZERO = True
	if ALL_ZERO:
		pos_ctrl_param.k1 = np.zeros(3)
		pos_ctrl_param.k2 = np.zeros(3)
		pos_ctrl_param.gamma = np.zeros(3)
		pos_ctrl_param.lmd = np.zeros(3)
	else:
		pos_ctrl_param.k1 = np.random.random(3)
		pos_ctrl_param.k2 = np.random.random(3)
		pos_ctrl_param.gamma = np.random.random() * np.ones(3)
		pos_ctrl_param.lmd = np.random.random() * np.ones(3)
	# pos_ctrl_param.print_param()
	'''随机初始化位置控制参数'''

	env = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)

	if TRAIN:
		action_std_init = 0.6
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
					K_epochs=50,
					eps_clip=0.2,
					action_std_init=action_std_init,
					buffer_size=int(env.time_max / env.dt * 2),  # 假设可以包含两条完整的最长时间的轨迹
					policy=policy,
					policy_old=policy_old,
					path=simulationPath)
		agent.PPO_info()
		max_training_timestep = int(env.time_max / env.dt) * 1000  # 10000回合
		action_std_decay_freq = int(9e6)
		action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
		min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

		sumr = 0
		start_eps = 0
		train_num = 0
		test_num = 0
		index = 0

		while timestep <= max_training_timestep:
			'''这整个是一个 episode 的初始化过程'''
			env.reset_random()
			env.show_image(False)
			sumr = 0.
			'''这整个是一个 episode 的初始化过程'''
			# t1 = time.time()
			while not env.is_terminal:
				env.current_state = env.next_state.copy()
				action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)  # 返回三个没有梯度的tensor
				env.get_param_from_actor(action_from_actor.detach().cpu().numpy().flatten())  # 将控制器参数更新
				action_4_uav = env.generate_action_4_uav()
				env.step_update(action_4_uav)  # 环境更新的action需要是物理的action

				# if agent.episode % 50 == 0:  # 50 个回合测试一下看看
				# 	env.image = env.image_copy.copy()
				# 	env.draw_3d_points_projection(np.atleast_2d([env.uav_pos(), env.pos_ref]), [Color().Red, Color().DarkGreen])
				# 	env.draw_time_error(env.uav_pos(), env.pos_ref)
				# 	env.show_image(False)

				sumr += env.reward
				agent.buffer.append(s=env.current_state,
									a=action_from_actor,  # .cpu().numpy()
									log_prob=a_log_prob.numpy(),
									r=env.reward,
									sv=s_value.numpy(),
									done=1.0 if env.is_terminal else 0.0,
									index=index)
				index += 1
				timestep += 1
				if timestep % action_std_decay_freq == 0:
					agent.decay_action_std(action_std_decay_rate, min_action_std)

				'''经验池填满的时候，开始新的一次学习'''
				if timestep % agent.buffer.batch_size == 0:
					print('  ~~~~~~~~~~ LEARN ~~~~~~~~~~')
					print('  Episode: {}'.format(agent.episode))
					print('  Num of learning: {}'.format(train_num))
					agent.learn()

					train_num += 1					# 训练次数加一
					start_eps = agent.episode		# PPO 中缺省
					# sumr = 0
					index = 0						# 用于记录数据轨迹的索引，学习一次后，经验池清零，索引归零
					print('  ~~~~~~~~~~ LEARN ~~~~~~~~~~')

					'''每学习 50 次，保存一下'''
					if train_num % 50 == 0 and train_num > 0:
						# 	average_test_r = agent.agent_evaluate(5)
						test_num += 1
						print('  Training count: {}...check point save...'.format(train_num))
						temp = simulationPath + 'episode_{}_trainNum_{}/'.format(agent.episode, train_num)
						os.mkdir(temp)
						time.sleep(0.01)
						agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
					'''每学习 50 次，保存一下'''
				'''经验池填满的时候，开始新的一次学习'''

			print('Episode: %d | Sumr: %.2f' % (agent.episode , sumr))

			if agent.episode % 10 == 0:
				test_num = 1
				print('TRAINING PAUSE......')
				print('Testing...')
				for i in range(test_num):
					env.reset_random()
					env.show_image(False)
					sumr = 0.
					while not env.is_terminal:
						_action_from_actor = agent.evaluate(env.current_state)
						env.get_param_from_actor(_action_from_actor.detach().cpu().numpy().flatten())  # 将控制器参数更新
						_action_4_uav = env.generate_action_4_uav()
						env.step_update(_action_4_uav)
						sumr += env.reward
						env.image = env.image_copy.copy()
						env.draw_3d_points_projection(np.atleast_2d([env.uav_pos(), env.pos_ref]), [Color().Red, Color().DarkGreen])
						env.draw_time_error(env.uav_pos(), env.pos_ref)
						env.show_image(False)
					print('Test ', i, ' reward: ', sumr)
					test_episode.append(agent.episode)
					test_reward.append(sumr)
				print('Testing Finished')
				print('TRAINING Continue......')
				pd.DataFrame({'episode': test_episode, 'reward': test_reward}).to_csv(simulationPath + 'test_record.csv')
			agent.episode += 1
	else:
		pass
