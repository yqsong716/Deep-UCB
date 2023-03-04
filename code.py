import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym, random, math
from gym.wrappers import frame_stack, atari_preprocessing

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# set seed
setup_seed(1)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7*7*64, 512)
        self.fc5 = nn.Linear(512, N_ACTIONS)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)

class Memory(object):
    def __init__(self, memory_size):
        self.memory = []
        self.memory_size = memory_size
        self.next_idx = 0

    def push(self, s, a, r, s_, done):

        data = (s, a, r, s_, done)
        if len(self.memory) <= self.memory_size:
            self.memory.append(data)
        else:
            self.memory[self.next_idx] = data
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def size(self):
        return len(self.memory)

class DQN(object):
    def __init__(self, n_actions, memory_size=1000, batch_size=32, lr=0.0001, GAMMA=0.99):
        self.eval_net, self.target_net = Net(), Net()
        self.target_net.load_state_dict(self.eval_net.state_dict())
        if USE_CUDA:
            self.eval_net = self.eval_net.cuda()
            self.target_net = self.target_net.cuda()
        self.memory = Memory(memory_size)
        self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=lr, eps=0.001, alpha=0.95)
        self.batch_size = batch_size
        self.gamma = GAMMA
        self.n_actions = n_actions

        self.UCB_net = Net()
        if USE_CUDA:
            self.UCB_net = self.UCB_net.cuda()
        self.UCB_optimizer = torch.optim.RMSprop(self.UCB_net.parameters(), lr=lr, eps=0.001, alpha=0.95)

    def choose_action(self, x, t):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        actions_value = self.eval_net.forward(x.cuda())
        UCB_value = self.UCB_net.forward(x.cuda())
        alpha = Alpha(t)
        value = self.fun(actions_value, UCB_value, alpha)
        action = torch.max(value.view(1, -1), 1)[1].numpy()
        action = action[0]
        if np.random.uniform() < alpha:
            self.UCB_learn(UCB_value)
        return action

    def compute_td_loss(self, states, actions, rewards, next_states, is_dones):
        b_s = torch.FloatTensor(states)
        b_a = torch.LongTensor(actions).view(BATCH_SIZE, 1)
        b_r = torch.FloatTensor(rewards).view(BATCH_SIZE, 1)
        b_s_ = torch.FloatTensor(next_states)
        b_d = torch.tensor(is_dones).bool().view(BATCH_SIZE, 1)

        if USE_CUDA:
            b_s = b_s.cuda()
            b_a = b_a.cuda()
            b_r = b_r.cuda()
            b_s_ = b_s_.cuda()
            b_d = b_d.cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_.view(BATCH_SIZE, 4, 84, 84))
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        q_target = torch.where(b_d, b_r, q_target)

        loss = F.smooth_l1_loss(q_eval, q_target.detach())
        return loss

    def sample(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in range(self.batch_size):
            idx = random.randint(0, self.memory.size() - 1)
            data = self.memory.memory[idx]
            state, action, reward, next_state, done = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return states, actions, rewards, next_states, dones

    def learn(self):
        states, actions, rewards, next_states, dones = self.sample()
        td_loss = self.compute_td_loss(states, actions, rewards, next_states, dones)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
        return td_loss.item()

    def UCB_learn(self, UCB_value):
        new_value = UCB_value.clone().detach()
        idx = torch.argsort(UCB_value)
        for i in range(self.n_actions):
            #tmp = 5 - torch.where(idx==i)[1]
            #tmp = idx[0][tmp[0].item()]
            new_value[0][i] = UCB_value[0][idx[0][(5 - torch.where(idx == i)[1])[0].item()]]
        UCB_loss = F.smooth_l1_loss(UCB_value, new_value)
        self.UCB_optimizer.zero_grad()
        UCB_loss.backward()
        self.UCB_optimizer.step()
        return UCB_loss.item()

    def fun(self, x, u, alpha):
        x = x.sigmoid().view(-1, 1)
        u = u.sigmoid().view(-1, 1)
        value = x * (1-alpha) + u * alpha
        return value.cpu()

env = gym.make('PongNoFrameskip-v4')
env = atari_preprocessing.AtariPreprocessing(env)
env = frame_stack.FrameStack(env, 4)

USE_CUDA = torch.cuda.is_available()

BATCH_SIZE = 32
LR = 0.0001                # learning rate
frames = 1200000
base = 30000
Alpha = lambda t: 0.05 + (1 - 0.05) * math.exp(
            -1. * t / base)

GAMMA = 0.99                 # reward discount
TARGET_REPLACE_ITER = 1000  # target update frequency
MEMORY_CAPACITY = 100000
print_interval = 1000

N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[2]
in_channels = 4

all_rewards = []
losses = []

episode_reward = 0
episode_num = 0

dqn = DQN(N_ACTIONS, MEMORY_CAPACITY, BATCH_SIZE, LR, GAMMA)
observation = env.reset()
for i in range(1, frames):

    env.render()
    a = dqn.choose_action(observation, i)
    observation_, r, done, info = env.step(a)
    episode_reward += r
    dqn.memory.push(observation, a, r, observation_, done)
    observation = observation_

    loss = 0
    if dqn.memory.size() > 10000:
        loss = dqn.learn()
        losses.append(loss)

    if i % print_interval == 0:
        localtime = time.asctime(time.localtime(time.time()))
        print('Frames:', i, '|', 'average_reward %5f' % np.mean(all_rewards[-10:]),
              '|', 'loss %5f' % loss,
              '|', 'Episode:', episode_num, '|', 'time', localtime)

    if i % TARGET_REPLACE_ITER == 0:
        dqn.target_net.load_state_dict(dqn.eval_net.state_dict())

    if done:
        observation = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        episode_num += 1

print('learning finished！')
