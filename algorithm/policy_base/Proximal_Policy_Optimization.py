import torch

from common.common_cls import *
import cv2 as cv
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F

"""use CPU or GPU"""
use_cuda = torch.cuda.is_available()
use_cpu_only = True
device = torch.device("cpu") if use_cpu_only else torch.device("cuda" if use_cuda else "cpu")
"""use CPU or GPU"""


def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class PPOActor_Gaussian(nn.Module):
    def __init__(self,
                 state_dim=3,
                 action_dim=3,
                 a_min=np.zeros(3),
                 a_max=np.ones(3),
                 use_orthogonal_init: bool = True):
        super(PPOActor_Gaussian, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = nn.Tanh()
        self.a_min = torch.tensor(a_min)
        self.a_max = torch.tensor(a_max)
        self.off = (self.a_min + self.a_max) / 2.0
        self.gain = self.a_max - self.off
        # self.std = 0.7

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = torch.tanh(self.mean_layer(s)) * self.gain + self.off
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        # dist = Normal(mean, self.std)
        return dist


class PPOCritic(nn.Module):
    def __init__(self, state_dim=3, use_orthogonal_init: bool = True):
        super(PPOCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.activate_func = nn.Tanh()

        if use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class Proximal_Policy_Optimization:
    def __init__(self,
                 env,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 gamma: float = 0.99,
                 K_epochs: int = 10,
                 eps_clip: float = 0.2,
                 action_std_init: float = 0.6,
                 buffer_size: int = 1200,
                 max_train_steps: int = int(5e6),
                 actor: PPOActor_Gaussian = PPOActor_Gaussian(),
                 critic: PPOCritic = PPOCritic(),
                 path: str = ''):
        """
        @param env:                 the environment
        @param actor_lr:            actor learning rate
        @param critic_lr:           critic learning rate
        @param gamma:               discount factor
        @param K_epochs:            training times for each leanr()
        @param eps_clip:            PPO clip for net update
        @param action_std_init:     exploration std
        @param buffer_size:         buffer size
        @param actor:               actor net
        @param critic:              critic net
        @param path:                path for data record
        """
        self.env = env
        '''PPO'''
        self.gamma = gamma  # discount factor
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.action_std = action_std_init
        self.path = path
        self.buffer = RolloutBuffer(buffer_size, self.env.state_dim, self.env.action_dim)
        # self.buffer2 = RolloutBuffer2(self.env.state_dim, self.env.action_dim)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        '''PPO'''

        '''Trick params'''
        self.set_adam_eps = True        # PPO recommends eps to be 1e-5
        self.lamda = 0.95               # gae param
        self.use_adv_norm = True        # advantage normalization
        self.mini_batch_size = 64       # 每次学习的时候，从大 batch 里面取的数
        self.entropy_coef = 0.01        # PPO Entropy coefficient
        self.use_grad_clip = True       # use_grad_clip
        self.use_lr_decay = True
        self.max_train_steps = max_train_steps
        self.using_mini_batch = False

        '''networks'''
        self.actor = actor
        self.critic = critic

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.loss = nn.MSELoss()
        self.device = device  # 建议使用 CPU 训练
        '''networks'''

        self.episode = 0
        self.reward = 0

    def evaluate(self, state):
        with torch.no_grad():
            t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
            action_mean = self.actor(t_state)
        return action_mean.detach().cpu().numpy().flatten()

    def choose_action(self, state: np.ndarray):
        with torch.no_grad():
            t_state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
            dist = self.actor.get_dist(t_state)
            a = dist.sample()  # Sample the action according to the probability distribution
            a = torch.maximum(torch.minimum(a, self.actor.a_max), self.actor.a_min)     # bounded to [min, max]
            a_logprob = dist.log_prob(a)  # The log probability density of the action
        return a.detach().cpu().numpy().flatten(), a_logprob.detach().cpu().numpy().flatten()

    def learn(self, current_steps):
        """
        @note: 	 network update
        @return: None
        """
        '''前期数据处理'''
        s, a, a_lp, r, s_, done, success = self.buffer.to_tensor()
        adv = []
        gae = 0.
        with torch.no_grad():
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - success) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        if self.using_mini_batch:
            for _ in range(self.K_epochs):      # 每次的轨迹数据学习 K_epochs 次
                for index in BatchSampler(SubsetRandomSampler(range(self.buffer.batch_size)), self.mini_batch_size, False):
                    dist_now = self.actor.get_dist(s[index])
                    dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                    a_logprob_now = dist_now.log_prob(a[index])

                    # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                    ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_lp[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                    surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv[index]
                    actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                    # Update actor
                    self.optimizer_actor.zero_grad()
                    actor_loss.mean().backward()

                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.optimizer_actor.step()

                    v_s = self.critic(s[index])
                    critic_loss = F.mse_loss(v_target[index], v_s)

                    # Update critic
                    self.optimizer_critic.zero_grad()
                    critic_loss.backward()
                    if self.use_grad_clip:  # Trick 7: Gradient clip
                        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.optimizer_critic.step()
        else:
            for _ in range(self.K_epochs):      # 每次的轨迹数据学习 K_epochs 次
                dist_now = self.actor.get_dist(s)
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a)

                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_lp.sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * adv  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()

                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s)
                critic_loss = F.mse_loss(v_target, v_s)

                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(current_steps)

    def lr_decay(self, total_steps):
        if total_steps < self.max_train_steps:
            lr_a_now = self.actor_lr * (1 - total_steps / self.max_train_steps)
            lr_c_now = self.critic_lr * (1 - total_steps / self.max_train_steps)
            lr_a_now = max(lr_a_now, 1e-6)
            lr_c_now = max(lr_c_now, 1e-6)
            for p in self.optimizer_actor.param_groups:
                p['lr'] = lr_a_now
            for p in self.optimizer_critic.param_groups:
                p['lr'] = lr_c_now

    def PPO_info(self):
        print('agent name：', self.env.name)
        print('state_dim:', self.env.state_dim)
        print('action_dim:', self.env.action_dim)
        print('action_range:', self.env.action_range)

    def action_linear_trans(self, action):
        # the action output
        linear_action = []
        for i in range(self.env.action_dim):
            a = min(max(action[i], -1), 1)
            maxa = self.env.action_range[i][1]
            mina = self.env.action_range[i][0]
            k = (maxa - mina) / 2
            b = (maxa + mina) / 2
            linear_action.append(k * a + b)
        return np.array(linear_action)

    def save_ac(self, msg, path):
        torch.save(self.actor.state_dict(), path + 'actor_' + msg)
        torch.save(self.critic.state_dict(), path + 'critic_' + msg)
