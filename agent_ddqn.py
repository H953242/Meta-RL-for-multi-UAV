import torch                                    # 导入torch
import torch.nn as nn                           # 导入torch.nn
import torch.nn.functional as F                 # 导入torch.nn.functional
import numpy as np                              # 导入numpy
from copy import deepcopy


# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self, agent_id, args):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()
        self.input_size = args.state_dim + args.action_dim
        self.fc1 = nn.Linear(self.input_size, 50)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(50, args.n_actions)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):     # 定义forward函数 (x为状态)

        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值


# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self,  agent_id, args):  # 定义DQN的一系列属性
        self.epsilon = args.epsilon
        self.beta = args.beta
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau  # 更新目标网络的周期
        self.target_lr = 0.01  # 目标网络更新的学习率
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.n_actions = args.n_actions  # 动作数
        self.eval_net, self.target_net = Net(agent_id, args), Net(agent_id, args)                # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # 用于 target 更新计时
        self.memory_counter = 0                                                 # 记忆库记数
        self.memory = np.zeros((self.buffer_size, self.state_dim * 2 + self.action_dim * 2 + 1))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)
        self.loss_save = []

    def choose_action(self, state, mean_action):    # 玻尔兹曼策略
        #action_probs_numes = []
        #denom = 0
        input_state = np.append(state, mean_action)  # input_state是一维向量
        x = torch.unsqueeze(torch.FloatTensor(input_state), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        actions_value = self.eval_net.forward(x).data.numpy()[0]                            # 通过对评估网络输入状态x，前向传播获得动作值,并转为numpy数组
        #actions_value = self.eval_net.forward(x)[0]
        print("ac", actions_value)
        # 改进版softmax
        c = np.max(actions_value)
        exp_a = np.exp((actions_value-c)/self.beta)
        sum_exp_a = np.sum(exp_a)
        action_probs = exp_a/sum_exp_a

        #action_probs = F.softmax(actions_value).data.numpy()
        print("softmax策略为：", action_probs)
        return np.random.choice(np.arange(self.n_actions), 1, p=action_probs)[0]

    def store_transition(self, s, a, r, s_, mean_a):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_, [mean_a]))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % self.buffer_size                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def get_q_target(self, reward, q_next):
        if np.random.uniform() > self.epsilon:                                  # 生成一个在[0, 1)内的随机数，如果大于EPSILON，选择最优动作
            v_mf = q_next.max(1)[0].view(self.batch_size, 1)                                                  # 输出action的第一个数
        else:
            random_index = np.random.randint(self.n_actions, size=(self.batch_size, self.action_dim), dtype="int64")# 随机选择动作
            index = torch.as_tensor(random_index)
            v_mf = q_next.gather(1, index)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
        return reward + self.gamma * v_mf                                                          # 返回选择的动作 (0或1)

    def learn(self):                                                            # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % self.tau == 0:                  # 一开始触发，然后每1步触发
            target_params = self.target_net.state_dict()
            behaviour_params = self.eval_net.state_dict()
            new_params = behaviour_params  # temp
            for k, v in behaviour_params.items():
                new_params[k] = (self.target_lr * v) + ((1 - self.target_lr) * target_params[k])
            self.target_net.load_state_dict(new_params)         # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1                                            # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(self.buffer_size, self.batch_size)       # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]                                 # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :self.state_dim])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, self.state_dim:self.state_dim+self.action_dim].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, self.state_dim+self.action_dim:self.state_dim+self.action_dim+1])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, self.state_dim+self.action_dim+1:self.state_dim*2+self.action_dim+1])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列
        b_mean_a = torch.LongTensor(b_memory[:, (self.state_dim*2+self.action_dim+1):].astype(int))
        # 将32个mean_a抽出，转为64-bit integer (signed)形式，并存储到b_mean_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_mean_a为32行1列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        eval_input = torch.cat((b_s, b_mean_a), dim=1)  # 将状态和平均动作拼成一个新的张量
        q_eval = self.eval_net(eval_input).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        target_input = torch.cat((b_s_, b_mean_a), dim=1)  # 将状态和平均动作拼成一个新的张量
        max_a = self.eval_net(target_input).argmax(dim=1).view(self.batch_size, 1)
        # 用评估网络选择出s‘下Q值最大的动作
        q_next = self.target_net(target_input).gather(1, max_a)
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + self.gamma * q_next
        #y = self.get_q_target(b_r, q_next)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        loss_np = loss.detach().numpy()
        self.loss_save.append(loss_np)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数
