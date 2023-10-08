import os
import sys
import datetime
import time
import cv2 as cv

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from environment.envs.UAV.uav_pos_ctrl_RL import uav_pos_ctrl_RL, uav_param
from environment.envs.UAV.FNTSMC import fntsmc_param
from environment.envs.UAV.ref_cmd import *
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
uav_param.time_max = 60
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
pos_ctrl_param.k1 = np.array([0.6, 0.4, 0.25])			# 1.2, 0.8, 0.5
pos_ctrl_param.k2 = np.array([0.1, 0.3, 0.25])			# 0.2, 0.6, 0.5
pos_ctrl_param.alpha = np.array([1.2, 1.5, 1.2])
pos_ctrl_param.beta = np.array([0.3, 0.3, 0.5])
pos_ctrl_param.gamma = np.array([0., 0., 0.])		# 0.2
pos_ctrl_param.lmd = np.array([0., 0., 0.])			# 2.0
pos_ctrl_param.dim = 3
pos_ctrl_param.dt = DT
pos_ctrl_param.ctrl0 = np.array([0., 0., 0.])
pos_ctrl_param.saturation = np.array([np.inf, np.inf, np.inf])
'''Parameter list of the position controller'''

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


setup_seed(3407)


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
			nn.ReLU()		# 因为是参数优化，所以最后一层用ReLU
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
		action_mean = self.actor(_s)		# PPO 给出的是分布，所以这里直接拿出的只能是 mean
		cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)		# 协方差矩阵
		dist = MultivariateNormal(action_mean, cov_mat)

		_a = dist.sample()
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
	log_dir = '../../../datasave/log/'
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	simulationPath = log_dir + datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H-%M-%S') + '-' + ALGORITHM + '-' + ENV + '/'
	os.mkdir(simulationPath)
	c = cv.waitKey(1)
	TRAIN = True  # 直接训练
	RETRAIN = False  # 基于之前的训练结果重新训练
	TEST = not TRAIN

	env = uav_pos_ctrl_RL(uav_param, att_ctrl_param, pos_ctrl_param)

	if TRAIN:
		action_std_init = 0.8
		'''重新加载Policy网络结构，这是必须的操作'''
		policy = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy', simulationPath)
		policy_old = PPOActorCritic(env.state_dim, env.action_dim, action_std_init, 'Policy_old', simulationPath)
		# optimizer = torch.optim.Adam([
		# 	{'params': agent.policy.actor.parameters(), 'lr': agent.actor_lr},
		# 	{'params': agent.policy.critic.parameters(), 'lr': agent.critic_lr}
		# ])
		'''重新加载Policy网络结构，这是必须的操作'''
		agent = PPO(env=env,
					actor_lr=3e-4,
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
		max_training_timestep = int(env.time_max / env.dt) * 10000  # 10000回合
		action_std_decay_freq = int(2.5e5)
		action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
		min_action_std = 0.1  # minimum action_std (stop decay after action_std <= min_action_std)

		sumr = 0
		start_eps = 0
		train_num = 0
		test_num = 0
		index = 0
		while timestep <= max_training_timestep:
			'''这整个是一个 episode 的初始化过程'''
			ref_amplitude, ref_period, ref_bias_a, ref_bias_phase = env.generate_random_circle(yaw_fixed=False)
			ref, _, _, _ = ref_uav(0., ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)

			phi_d = phi_d_old = 0.
			theta_d = theta_d_old = 0.
			dot_phi_d = (phi_d - phi_d_old) / env.dt
			dot_theta_d = (theta_d - theta_d_old) / env.dt
			throttle = env.m * env.g

			uav_param.time_max = 30
			uav_param.pos0 = env.set_random_init_pos(pos0=ref[0:3], r=0.3 * np.ones(3))

			env.uav_reset_with_new_param(new_uav_param=uav_param)  # 无人机初始参数，只变了初始位置
			env.controller_reset_with_new_param(new_att_param=att_ctrl_param, new_pos_param=pos_ctrl_param)  # 控制器参数，一般不变
			env.collector_reset(round(uav_param.time_max / uav_param.dt))
			ref_traj = env.generate_ref_pos_trajectory(ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)
			env.draw_3d_trajectory_projection(ref_traj)
			env.draw_init_image()
			env.draw_3d_points_projection(np.atleast_2d([env.uav_pos(), ref[0: 3]]), [Color().Red, Color().DarkGreen])
			env.show_image(True)
			'''这整个是一个 episode 的初始化过程'''

			while not env.is_terminal:
				env.current_state = env.next_state.copy()
				action_from_actor, s, a_log_prob, s_value = agent.choose_action(env.current_state)  # 返回三个没有梯度的tensor
				action_from_actor = action_from_actor.numpy()	# 这里的 action 实际上就是真是的参数，需要把它们复制给控制器
				env.get_param_from_actor(action_from_actor)		# 将控制器参数更新

				'''控制'''
				'''3.1 generate '''
				ref, dot_ref, dot2_ref, _ = ref_uav(env.time, ref_amplitude, ref_period, ref_bias_a, ref_bias_phase)  # xd yd zd psid
				uncertainty = generate_uncertainty(time=env.time, is_ideal=True)
				obs = np.zeros(3)

				'''3.2 outer-loop control'''
				phi_d_old = phi_d
				theta_d_old = theta_d
				phi_d, theta_d, throttle = env.pos_control(ref[0:3], dot_ref[0:3], dot2_ref[0:3], uncertainty, obs)
				dot_phi_d = (phi_d - phi_d_old) / env.dt
				dot_theta_d = (theta_d - theta_d_old) / env.dt

				'''3.3 inner-loop control'''
				rho_d = np.array([phi_d, theta_d, ref[3]])
				dot_rho_d = np.array([dot_phi_d, dot_theta_d, dot_ref[3]])
				torque = env.att_control(rho_d, dot_rho_d, np.zeros(3), att_only=False)
				action_4_uav = [throttle, torque[0], torque[1], torque[2]]

				'''3.4 step update'''
				env.step_update(action_4_uav)  # 环境更新的action需要是物理的action

				if agent.episode % 50 == 0:		# 50 个回合测试一下看看
					env.image = env.image_copy.copy()
					env.draw_3d_points_projection(np.atleast_2d([env.uav_pos(), ref[0: 3]]), [Color().Red, Color().DarkGreen])
					env.draw_error(env.uav_pos(), ref[0:3])
					env.show_image(False)
				'''控制'''

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
				'''学习'''
				if timestep % agent.buffer.batch_size == 0:
					print('========== LEARN ==========')
					print('Episode: {}'.format(agent.episode))
					print('Num of learning: {}'.format(train_num))
					agent.learn()
					'''clear buffer'''
					average_train_r = round(sumr / (agent.episode + 1 - start_eps), 3)
					print('Average reward:', average_train_r)
					# agent.writer.add_scalar('train_r', average_train_r, train_num)		# to tensorboard
					train_num += 1
					start_eps = agent.episode
					sumr = 0
					index = 0
					if train_num % 50 == 0 and train_num > 0:
						average_test_r = agent.agent_evaluate(5)
						# agent.writer.add_scalar('test_r', average_test_r, test_num)	# to tensorboard
						test_num += 1
						print('check point save')
						temp = simulationPath + 'episode' + '_' + str(agent.episode) + '_save/'
						os.mkdir(temp)
						time.sleep(0.01)
						agent.policy_old.save_checkpoint(name='Policy_PPO', path=temp, num=timestep)
					print('========== LEARN ==========')
				'''学习'''



				if timestep % action_std_decay_freq == 0:
					agent.decay_action_std(action_std_decay_rate, min_action_std)
			agent.episode += 1
	else:
		pass
