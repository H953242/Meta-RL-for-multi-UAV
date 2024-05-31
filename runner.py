from pylab import *
from agent import DQN
from final_env_100u_meta import Environ
import time
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
mpl.rcParams['font.sans-serif'] = ['SimHei']


class Runner:
    def __init__(self, args, env):
        self.args = args
        self.epsilon = args.epsilon
        self.env = env
        self.possible_actions = np.arange(args.n_actions)  # 可选动作的集合
        self.agents = self._init_agents()
        self.reward_mean_for_episode = np.zeros([args.n_agent, args.episode])  # 记录所有智能体的回合平均奖励
        self.channel_selected = np.zeros([args.n_agent, args.episode, args.time_steps])  # 记录所有智能体的信道选择
        self.loss_save=[]
        self.throughput_save=[]
        #self.reward_mean_for_episode = np.zeros([args.n_agent, 2])
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agent):
            agent = DQN(i, self.args)
            agents.append(agent)
        return agents

    # def _init_(self):
    #     self.environ= Environ()
    #     throughput_save=[]
    #     throughput = Environ.get_reward()
    #     throughput_save.append(throughput)
    #     return throughput_save

    def run(self):
        for episode in tqdm(range(self.args.episode),desc='run process'):
        #for episode in range(2):
            s, mean_actions = self.env.reset()
            episode_reward_sum = np.zeros([self.env.n_agent, self.env.n_des], dtype=float)  # 初始化该循环对应的episode的总奖励
            for t in range(self.args.time_steps):
                # print("epi{}  step{}".format(episode, t))
                actions = []  # 记录本时刻的联合动作，每个step都重置
                # 每个智能体各自选择动作
                for agent_id, agent in enumerate(self.agents):
                    action = agent.choose_action(s[agent_id*3],s[agent_id*3+1],s[agent_id*3+2], mean_actions[agent_id],)
                    actions.append(action)
                    self.channel_selected[agent_id, episode, t] = action
                # 产生新状态和reward
                s_next, mean_actions, r, done, info = self.env.step(actions)
                episode_reward_sum += r  # 逐步加上一个episode内每个step的reward
                # 计算各自的平均动作，向经验池存入数据，记录平均动作用于下一时刻动作选择，智能体学习一次
                for agent_id, agent in enumerate(self.agents):
                    agent.store_transition(s[agent_id*3],s[agent_id*3+1],s[agent_id*3+2], actions[agent_id], r[agent_id], s_next[agent_id*3],s_next[agent_id*3+1],s_next[agent_id*3+2], mean_actions[agent_id])
                    if agent.memory_counter > agent.buffer_size:              # 如果累计的transition数量超过了记忆库的固定容量
                        loss = agent.learn()
                        self.loss_save.append(loss)


                s = s_next
            throughput = self.env.get_reward()
            self.throughput_save.append(throughput)
            for i in range(len(episode_reward_sum)):
                self.reward_mean_for_episode[i, episode] = round(episode_reward_sum[i, 0]/self.args.time_steps, 2)

        print("training_finished")

    def plot(self):
        # print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        # np.savetxt('mfq_100u40c_200e_2000s', self.reward_mean_for_episode)
        # np.savetxt('loss',self.loss_save)
        np.savetxt('throughput', self.throughput_save)
        np.save('mfq-ddqn_10u6c_200e_2000s_channel_selected', self.channel_selected)
        x = np.array(range(self.args.episode))
        plt.plot(x, self.reward_mean_for_episode[0, :], ls='-', color='red', label='0')
        plt.plot(x, self.reward_mean_for_episode[1, :], ls='-', color='blue', label='1')
        plt.plot(x, self.reward_mean_for_episode[2, :], ls='--', color='red', label='2')
        plt.plot(x, self.reward_mean_for_episode[3, :], ls='--', color='blue', label='3')
        plt.plot(x, self.reward_mean_for_episode[4, :], ls='-.', color='red', label='4')
        plt.plot(x, self.reward_mean_for_episode[5, :], ls='-.', color='blue', label='5')
        plt.legend()
        plt.xlabel("episode")
        plt.ylabel("mean reward for time_steps times in one episode")
        plt.title('各无人机的单回合平均奖励变化')
        # 保存图片到本地
        plt.savefig(self.save_path + '/2022_4_7_200e_2000s.png', format='png')
        plt.show()
