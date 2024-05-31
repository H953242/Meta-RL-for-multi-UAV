from __future__ import division
import numpy as np
import time
import random
import math
import gym
from gym import spaces
from copy import deepcopy
from threading import Timer
from datetime import datetime
from scipy.special import comb, perm
from itertools import combinations, permutations

#无人机，干扰机都和基站通信
'''
class UAVchannels:
    def __init__(self, n_uav, n_channel, BS_position):
        self.h_bs = 25  # BS antenna height
        self.h_uav = 1.5  # uav antenna height
        self.fc = 2  # 载频2GHz
        self.BS_position = BS_position
        self.n_uav = n_uav
        self.n_channel = n_channel
        # self.Decorrelation_distance = 50
        # self.shadow_std = 8  # 阴影标准偏差
        # self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1, d2)  # 返回欧几里德范数 sqrt(x*x + y*y)
            # change from meters to kilometers
            # Pathloss model : 103.8 + 20.9log10(R), R in kilometers
            self.PathLoss[i] = 103.8 + 20.9 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_uav) ** 2) / 1000)

    # def update_shadow(self, delta_distance_list):
    #     if len(delta_distance_list) == 0:  # 初始化Shadow
    #         self.Shadow = np.random.normal(0, self.shadow_std, self.n_uav)
    #     else:
    #         delta_distance = np.asarray(delta_distance_list)
    #         self.Shadow = np.exp(-1 * (delta_distance / self.Decorrelation_distance)) * self.Shadow + \
    #                       np.sqrt(1 - np.exp(- 2 * (delta_distance / self.Decorrelation_distance))) * \
    #                       np.random.normal(0, self.shadow_std, self.n_uav)

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_uav, self.n_channel)) +
                              1j * np.random.normal(size=(self.n_uav, self.n_channel)))
        self.FastFading = 20 * np.log10(np.abs(h))


class Jammerchannels:
    def __init__(self, n_jammer, n_channel, BS_position):
        self.h_bs = 25  # BS antenna height
        self.h_jammer = 1.5  # jammer antenna height
        self.BS_position = BS_position
        self.n_jammer = n_jammer
        self.n_channel = n_channel
        # self.Decorrelation_distance = 50
        # self.shadow_std = 8  # 阴影标准偏差
        # self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1, d2)  # 返回欧几里德范数 sqrt(x*x + y*y)
            # change from meters to kilometers
            # Pathloss model : 103.8 + 20.9log10(R), R in kilometers
            self.PathLoss[i] = 103.8 + 20.9 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_jammer) ** 2) / 1000)

    # def update_shadow(self, delta_distance_list):
    #     if len(delta_distance_list) == 0:  # 初始化Shadow
    #         self.Shadow = np.random.normal(0, self.shadow_std, self.n_jammer)
    #     else:
    #         delta_distance = np.asarray(delta_distance_list)
    #         self.Shadow = np.exp(-1 * (delta_distance / self.Decorrelation_distance)) * self.Shadow + \
    #                       np.sqrt(1 - np.exp(- 2 * (delta_distance / self.Decorrelation_distance))) * \
    #                       np.random.normal(0, self.shadow_std, self.n_jammer)

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_jammer, self.n_channel)) +
                              1j * np.random.normal(size=(self.n_jammer, self.n_channel)))
        self.FastFading = 20 * np.log10(np.abs(h))
'''


#无人机，干扰机都和无人机通信, 无人机之间不交互Q表独立,接收机与发射机之间有距离限制
class UAVchannels:
    def __init__(self, n_uav, n_channel, BS_position):
        self.h_bs = 25  # BS antenna height
        self.h_uav = 1.5  # uav antenna height
        self.fc = 2  # 载频2GHz
        self.BS_position = BS_position
        self.n_uav = n_uav
        self.n_channel = n_channel
        # self.Decorrelation_distance = 50
        # self.shadow_std = 8  # 阴影标准偏差
        # self.update_shadow([])

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions), len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d3 = abs(position_A[2] - position_B[2])
        distance = np.sqrt(d1**2 + d2**2 + d3**2) + 0.001
        PL_los = 103.8 + 20.9*np.log10(distance*1e-3)
        return PL_los

        # d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        # def PL_Los(d):
        #     if d <= 3:  # 如果距离小于3米，则使用3米的路径损耗。
        #         return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
        #     else:
        #         if d < d_bp:
        #             return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
        #         else:
        #             return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - \
        #                    17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)
        #
        # def PL_NLos(d_a, d_b):
        #     n_j = max(2.8 - 0.0024 * d_a, 1.84)
        #     return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)
        #
        # if min(d1, d2) < 7:
        #     PL = PL_Los(d)
        #     self.ifLOS = True
        #     self.shadow_std = 3
        # else:
        #     PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        #     self.ifLOS = False
        #     self.shadow_std = 4  # if Non line of sight, the std is 4
        # return PL

    # def update_shadow(self, delta_distance_list):  # 其中D（i，j）是从时间n-1到时间n的链路i到j的距离变化的更新距离矩阵。
    #     delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
    #     for i in range(len(delta_distance)):
    #         for j in range(len(delta_distance)):
    #             delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
    #     if len(delta_distance_list) == 0:  # 初始化Shadow
    #         self.Shadow = np.random.normal(0, self.shadow_std, size=(self.n_uav, self.n_uav))
    #     else:  # S(n) = exp(-D/D_corr) .* S(n-1) + sqrt{ (1-exp(-2*D/D_corr))}.*N_S(n)
    #         self.Shadow = np.exp(-1 * (delta_distance / self.decorrelation_distance)) * self.Shadow + \
    #                       np.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * \
    #                       np.random.normal(0, self.shadow_std, size=(self.n_uav, self.n_uav))

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_uav, self.n_uav, self.n_channel)) + 1j *
                              np.random.normal(size=(self.n_uav, self.n_uav, self.n_channel)))
        self.FastFading = 20 * np.log10(np.abs(h))


class Jammerchannels:
    def __init__(self, n_jammer, n_uav, n_channel, BS_position):
        self.h_bs = 25  # BS antenna height
        self.h_jammer = 1.5  # jammer antenna height
        self.h_uav = 1.5  # uav antenna height
        self.BS_position = BS_position
        self.n_jammer = n_jammer
        self.n_uav = n_uav
        self.n_channel = n_channel
        # self.Decorrelation_distance = 50
        # self.shadow_std = 8  # 阴影标准偏差
        # self.update_shadow([])

    def update_positions(self, positions, uav_positions):
        self.positions = positions
        self.uav_positions = uav_positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions), len(self.uav_positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.uav_positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.uav_positions[j])

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d3 = abs(position_A[2] - position_B[2])
        distance = np.sqrt(d1**2 + d2**2 + d3**2) + 0.001
        PL_los = 103.8 + 20.9*np.log10(distance*1e-3)
        return PL_los

        # d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        # def PL_Los(d):
        #     if d <= 3:  # 如果距离小于3米，则使用3米的路径损耗。
        #         return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
        #     else:
        #         if d < d_bp:
        #             return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
        #         else:
        #             return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - \
        #                    17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)
        #
        # def PL_NLos(d_a, d_b):
        #     n_j = max(2.8 - 0.0024 * d_a, 1.84)
        #     return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)
        #
        # if min(d1, d2) < 7:
        #     PL = PL_Los(d)
        #     self.ifLOS = True
        #     self.shadow_std = 3
        # else:
        #     PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        #     self.ifLOS = False
        #     self.shadow_std = 4  # if Non line of sight, the std is 4
        # return PL

    # def update_pathloss(self):
    #     self.PathLoss = np.zeros(len(self.positions))
    #     for i in range(len(self.positions)):
    #         d1 = abs(self.positions[i][0] - self.BS_position[0])
    #         d2 = abs(self.positions[i][1] - self.BS_position[1])
    #         distance = math.hypot(d1, d2)  # 返回欧几里德范数 sqrt(x*x + y*y)
    #         # change from meters to kilometers
    #         # Pathloss model : 103.8 + 20.9log10(R), R in kilometers
    #         self.PathLoss[i] = 103.8 + 20.9 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_jammer) ** 2) / 1000)

    # def update_shadow(self, delta_distance_list):
    #     if len(delta_distance_list) == 0:  # 初始化Shadow
    #         self.Shadow = np.random.normal(0, self.shadow_std, self.n_jammer)
    #     else:
    #         delta_distance = np.asarray(delta_distance_list)
    #         self.Shadow = np.exp(-1 * (delta_distance / self.Decorrelation_distance)) * self.Shadow + \
    #                       np.sqrt(1 - np.exp(- 2 * (delta_distance / self.Decorrelation_distance))) * \
    #                       np.random.normal(0, self.shadow_std, self.n_jammer)
    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_jammer, self.n_uav, self.n_channel)) +
                              1j * np.random.normal(size=(self.n_jammer, self.n_uav, self.n_channel)))
        self.FastFading = 20 * np.log10(np.abs(h))


class UAV:
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []  # 记录邻居（发射机）的编号，只有发射机才有
        self.destinations = []  # 记录自己搭档的编号


class Jammer:
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity


class Environ(gym.Env):
    def __init__(self):
        self.seed_set()

        # self.backward_lanes = [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2,
        #                    750 - 3.5 - 3.5 / 2, 750 - 3.5 / 2]
        # self.forward_lanes = [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2,
        #                  500 + 3.5 + 3.5 / 2]

        # self.forward_lanes = [150, 350, 550, 750, 950]
        # self.backward_lanes = [50, 250, 450, 650, 850]
        # self.left_lanes = [75, 175, 275, 375, 475]
        # self.right_lanes = [25, 125, 225, 325, 425]
        self.forward_lanes = [75, 175, 275, 375, 475]
        self.backward_lanes = [25, 125, 225, 325, 425]
        self.left_lanes = [37.5, 87.5, 137.5, 187.5, 237.5]
        self.right_lanes = [12.5, 62.5, 112.5, 162.5, 212.5]
        # self.forward_lanes = [15, 35, 55, 75, 95]
        # self.backward_lanes = [5, 25, 45, 65, 85]
        # self.left_lanes = [7.5, 17.5, 27.5, 37.5, 47.5]
        # self.right_lanes = [2.5, 12.5, 22.5, 32.5, 42.5]
        # self.n_lanes = len(self.forward_lanes)
        self.length = 500  # 1000
        self.width = 250  # 500
        self.height = 80
        self.BS_position = [self.length / 2, self.width / 2, 0]  # Suppose the BS is in the center

        self.uav_power = 23  # dBm
        self.jammer_power = 23   # dBm
        self.sig2_dB = -114  # dBm       Noise power
        self.sig2 = 10 ** (self.sig2_dB / 10)
        # self.bsAntGain = 8  # dBi       BS antenna gain
        # self.bsNoiseFigure = 5  # dB      BS receiver noise figure
        self.uavAntGain = 3  # dBi       uav antenna gain
        self.uavNoiseFigure = 9  # dB    uav receiver noise figure
        self.jammerAntGain = 3  # dBi       jammer antenna gain
        # self.jammerNoiseFigure = 9  # dB    jammer receiver noise figure
        self.bandwidth = 1    #1.5 * 1e+6   # Hz
        self.uav_rate = 16   # bit每秒每赫兹

        self.t_Rx = 0.98   # 传输时间,单位都是s
        self.timestep = 0.2     # 频谱感知，选动作 + ACK + 学习
        self.timeslot = self.t_Rx + self.timestep    # 时隙
        self.t_uav = 0
        self.jammer_start = 0.2     # 干扰机开始干扰时间
        self.t_dwell = 2.28   # 干扰机扫频停留时间
        self.t_jammer = 0
        self.max_throughput = self.uav_rate * self.bandwidth * self.t_Rx

        self.n_transmitter = 10  # UAV发送机个数
        self.n_agent = self.n_transmitter
        self.n_receiver = self.n_transmitter  # UAV接收机个数
        self.n_des = 1  # 每个UAV发送机的通信目标数
        self.n_uav = self.n_transmitter + self.n_receiver  # number of UAVs
        self.n_uav_pair = self.n_transmitter * self.n_des  # 无人机对的数目
        self.n_jammer = 2  # number of jammers
        self.n_channel = 13  #int(self.n_transmitter+self.n_jammer-1)  # number of channels
        self.channel_indexes = np.arange(self.n_channel)  # 信道的标号，0到9的numpy一维数组
        self.channels = np.zeros([self.n_channel], dtype=int)  #
        self.p_md = 0   # 漏警概率
        self.p_fa = 0   # 虚警概率
        self.pn0 = 20  # 数据包长度
        self.states_observed = 2  # 信道被干扰或未被干扰

        self.n_step = 0
        self.max_distance = 50
        self.neightor_threshold = 100  # 用来判断邻居的阈值，相互间距离小于该值的发射机互为邻居
        self.is_jammer_moving = True
        self.type_of_interference = "markov"      #"markov"，"saopin"
        self.policy = None    # 对应算法
        self.training = True

        self.uav_list = list(np.arange(self.n_uav))  # 无人机的列表，0到15
        self.transmitter_list = random.sample(self.uav_list, k=self.n_transmitter)  # 从无人机列表里任选8个作为发射机
        self.uav_pairs = np.zeros([self.n_transmitter, self.n_des, 2], dtype=int)
        self.receiver_list = list(set(self.uav_list) - set(self.transmitter_list))

        self.action_range = int(perm(self.n_channel, self.n_des))  # 假设uav对通信，一个发送机若有多个通信目标，则用于和各个通信目标通信的信道不同
        self.all_actions_for_each_transmitter = list(permutations(self.channel_indexes, self.n_des))
        self.action_dim = self.n_des  # 现在无人机的动作就是信道选择，所以有几个通信对象动作就是几维的
        self.action_space = spaces.Discrete(self.action_dim)

        # self.all_observed_states()
        # self.reset()
        # self.state_dim = len(self.get_state())
        # self.observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.state_dim,))

        self.uav_state_dim = self.action_range
        self.all_observed_states()  # 获取无人机观测维度
        self.state_dim = self.n_channel + 1
        self.observation_space = spaces.Discrete(self.state_dim)
        # self.state_dim = (self.n_channel ** self.n_uav) * (self.states_observed ** self.n_channel)  # 漏警加虚警？？

    def seed_set(self, seed=2020):
        random.seed(seed)
        #os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        #tf.set_random_seed(seed)

    def all_observed_states(self):
        self.observed_state_list = []
        observed_state = 0
        self.all_observed_states_list = []
        if self.type_of_interference == "saopin":
            self.step_forward = 1
            # perm，智能体可以观测到当前某个干扰机是干扰的哪一个信道，没有虚警漏警的可能
            # self.observed_state_dim = int(perm(self.n_channel, self.n_jammer))
            # self.all_observed_states_list.extend(list(permutations(self.channel_indexes, self.n_jammer)))
            # comb,智能体只能感知信道是否被干扰，不知道某个干扰机干扰的是哪个信道，有虚警漏警的可能
            if self.p_md == 0 and self.p_fa == 0:
                self.observed_state_dim = int(comb(self.n_channel, self.n_jammer))
                self.all_observed_states_list.extend(list(combinations(self.channel_indexes, self.n_jammer)))
            elif self.p_md > 0 and self.p_fa == 0:  # 漏警
                for i in range(self.n_jammer + 1):
                    self.observed_state_dim += int(comb(self.n_channel, i))
                    self.all_observed_states_list.extend(list(combinations(self.channel_indexes, i)))
            elif self.p_md == 0 and self.p_fa > 0:  # 虚警
                for i in range(self.n_jammer, self.n_channel + 1):
                    self.observed_state_dim += int(comb(self.n_channel, i))
                    self.all_observed_states_list.extend(list(combinations(self.channel_indexes, i)))
        elif self.type_of_interference == "markov":
            # 随机生成一个干扰机干扰信道的转移概率矩阵，矩阵的横纵坐标是 self.cur_observed_states_list
            self.p_trans = np.random.uniform(0, self.n_channel, [self.n_channel, self.n_channel])
            p_trans_sum = np.sum(self.p_trans, axis=1)
            for i in range(self.n_channel):
                for j in range(self.n_channel):
                    self.p_trans[i][j] = self.p_trans[i][j]/p_trans_sum[i]
            # perm，智能体可以观测到当前某个干扰机是干扰的哪一个信道，没有虚警漏警的可能
            # self.observed_state_dim = self.p_trans.shape[0]
            # self.all_observed_states_list.extend(random.sample(list(permutations(self.channel_indexes, self.n_jammer)), k=self.observed_state_dim))
            # comb,智能体只能感知信道是否被干扰，不知道某个干扰机干扰的是哪个信道，有虚警漏警的可能
            if self.p_md == 0 and self.p_fa == 0:
                # self.observed_state_dim = self.p_trans.shape[0]
                # self.all_observed_states_list.extend(random.sample(list(combinations(self.channel_indexes, self.n_jammer)), k=self.observed_state_dim))
                if self.policy == "Q_learning":
                    self.observed_state_dim = int(comb(self.n_channel, self.n_jammer))
                    # all_observed_states_list是列表，其元素是元组
                    self.all_observed_states_list.extend(list(combinations(self.channel_indexes, self.n_jammer)))
                elif self.policy == "drl":
                    self.observed_state_dim = int(self.n_channel+1)

    def renew_uav_pairs(self):
        receiver_list = deepcopy(self.receiver_list)
        for i in range(self.n_transmitter):
            receivers = random.sample(receiver_list, k=self.n_des)
            for j in range(self.n_des):
                self.uav_pairs[i][j][0] = self.transmitter_list[i]
                self.uav_pairs[i][j][1] = receivers[j]
                self.uavs[self.transmitter_list[i]].destinations.append(receivers[j])
                self.uavs[receivers[j]].destinations.append(self.transmitter_list[i])

                tra_pos = [self.uavs[self.transmitter_list[i]].position[0], self.uavs[self.transmitter_list[i]].position[1]]
                rec_pos = [self.uavs[receivers[j]].position[0], self.uavs[receivers[j]].position[1]]
                dis = (rec_pos[0] - tra_pos[0]) ** 2 + (rec_pos[1] - tra_pos[1]) ** 2
                while dis > self.max_distance ** 2:
                    tra_xrange = [max(0, tra_pos[0]-self.max_distance), min(self.length, tra_pos[0]+self.max_distance)]
                    rec_pos[0] = random.uniform(tra_xrange[0], tra_xrange[1])
                    tra_yrange = [max(0, tra_pos[1] - self.max_distance), min(self.width, tra_pos[1] + self.max_distance)]
                    rec_pos[1] = random.uniform(tra_yrange[0], tra_yrange[1])
                    dis = (rec_pos[0] - tra_pos[0]) ** 2 + (rec_pos[1] - tra_pos[1]) ** 2

                self.uavs[receivers[j]].position[0] = rec_pos[0]
                self.uavs[receivers[j]].position[1] = rec_pos[1]
                self.uavs[receivers[j]].direction = None
                self.uavs[receivers[j]].velocity = None
            receiver_list = list(set(receiver_list) - set(receivers))   # 后期可改
        print(self.uav_pairs)

    def renew_uavs(self, n_uav):  # 初始化无人机位置
        all_directions = ["forward", "backward", "left", "right"]
        start_direction_list = []
        k = int(n_uav / 4)
        for _ in range(k):
            start_direction_list.extend(all_directions)
        m = int(n_uav % 4)
        start_direction_list.extend(random.sample(all_directions, k=m))

        for i in range(n_uav):
            if start_direction_list[i] == "forward":
                ind = np.random.randint(0, len(self.forward_lanes))
                start_position = [self.forward_lanes[ind], random.randint(0, self.width), self.height]

            elif start_direction_list[i] == "backward":
                ind = np.random.randint(0, len(self.backward_lanes))
                start_position = [self.backward_lanes[ind], random.randint(0, self.width), self.height]

            elif start_direction_list[i] == "left":
                ind = np.random.randint(0, len(self.left_lanes))
                start_position = [random.randint(0, self.length), self.left_lanes[ind], self.height]

            elif start_direction_list[i] == "right":
                ind = np.random.randint(0, len(self.right_lanes))
                start_position = [random.randint(0, self.length), self.right_lanes[ind], self.height]

            start_velocity = random.uniform(10, 20)
            self.uavs.append(UAV(start_position, start_direction_list[i], start_velocity))

        # self.V2V_Shadowing = np.random.normal(0, 3, [len(self.uavs), len(self.uavs)])  # 正态分布
        # self.V2I_Shadowing = np.random.normal(0, 8, len(self.uavs))  # 均值标准差
        # self.delta_distance = np.asarray([c.velocity for c in self.uavs])
        # self.renew_channel()

    def renew_jammers(self, n_jammer):  # 初始化干扰机位置
        start_velocity = 0
        all_directions = ["forward", "backward", "left", "right"]
        start_direction_list = []
        k = int(n_jammer / 4)
        for _ in range(k):
            start_direction_list.extend(all_directions)
        m = int(n_jammer % 4)
        start_direction_list.extend(random.sample(all_directions, k=m))

        for i in range(n_jammer):
            if start_direction_list[i] == "forward":
                ind = np.random.randint(0, len(self.forward_lanes))
                start_position = [self.forward_lanes[ind], random.randint(0, self.width), self.height]

            elif start_direction_list[i] == "backward":
                ind = np.random.randint(0, len(self.backward_lanes))
                start_position = [self.backward_lanes[ind], random.randint(0, self.width), self.height]

            elif start_direction_list[i] == "left":
                ind = np.random.randint(0, len(self.left_lanes))
                start_position = [random.randint(0, self.length), self.left_lanes[ind], self.height]

            elif start_direction_list[i] == "right":
                ind = np.random.randint(0, len(self.right_lanes))
                start_position = [random.randint(0, self.length), self.right_lanes[ind], self.height]

            if self.is_jammer_moving:
                start_velocity = random.uniform(10, 20)

            self.jammers.append(Jammer(start_position, start_direction_list[i], start_velocity))

    def renew_neighbors_of_uavs(self):
        for i in range(len(self.uavs)):
            self.uavs[i].neighbors = []
            # print('action and neighbors delete', self.uavs[i].actions, self.uavs[i].neighbors)
        Distance = np.zeros((len(self.uavs), len(self.uavs)))
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.uavs]])
        Distance = abs(z.T - z)
        for i in self.transmitter_list:
            for j in self.transmitter_list:
                if (i != j) and (Distance[i, j] <= self.neightor_threshold):
                    self.uavs[i].neighbors.append(j)
        # for i in range(len(self.uavs)):
        #     sort_idx = np.argsort(Distance[:, i])  # 返回数组值从小到大的索引值
        #     for j in range(len(sort_idx)-1):
        #         self.uavs[i].neighbors.append(sort_idx[j + 1])

    def new_random_game(self):  # 每次reset调用一次
        # self.all_observed_states()
        # 一个发送机若有多个通信目标，每个元素是智能体为每个通信目标分配的信道，假设各不相同
        self.uav_channels = np.zeros([self.n_transmitter], dtype='int32')   # 无人机记录各自选择的信道
        # 每个智能体观察到的全局动作（假设智能体可以观察到其他智能体已经完成的动作）
        #self.uav_channels = np.zeros([self.n_uav, self.n_uav], dtype='int32')
        for i in range(self.n_transmitter):  # 为所有无人机随机分配初始信道
            self.uav_channels[i] = random.randint(0, self.action_range - 1)  # 包括上下限， 在0到9（action_range - 1）之间随机返回一个整数
        # 干扰机选择初始干扰信道
        if self.type_of_interference == "saopin":
            self.jammer_channels = random.sample(range(0, self.n_channel), k=self.n_jammer)  #不包括 stop
        elif self.type_of_interference == "markov":
            # 将all_observed_states_list打乱顺序重排为 cur_observed_states_list，以此为转移概率矩阵的横纵坐标
            # 环境对象一旦建立转移概率矩阵就固定了，环境的每次reset对转移矩阵的坐标进行重排，以区别环境重启前后的干扰机
            if self.policy == "Q_learning":
                self.cur_observed_states_list = random.sample(self.all_observed_states_list, k=self.n_channel)
            elif self.policy == "drl":
                # 将所有的信道打乱顺序重排为 cur_observed_states_list，以此为转移概率矩阵的横纵坐标
                # 环境对象一旦建立转移概率矩阵就固定了，环境的每次reset对转移矩阵的坐标进行重排，以区别环境重启前后的干扰机
                self.cur_observed_states_list = []  # 放置所有干扰机的当前的状态观测表
                self.jammer_channels = []  # 记录干扰机所选信道
                for j in range(self.n_jammer):
                    cur_observed_states = random.sample(list(range(self.n_channel)), k=self.n_channel)
                    self.cur_observed_states_list.append(cur_observed_states)
                    jammer_j_channel = random.choices(self.cur_observed_states_list[j], k=1)[0]
                    self.jammer_channels.append(jammer_j_channel)
        self.jammer_channels_list = []
        self.jammer_index_list = []
        #如果传输阶段同时干扰两个信道,0是后半段 改变后的信道，1是前半段 改变前的信道
        self.jammer_time = np.zeros([2])  # 每个干扰机在传输阶段最多同时干扰两个信道，目前假设各个干扰机时间线相同
        # self.jammer_time = np.zeros([self.n_jammer, 2])
        # self.jammer_indexs = np.array(self.jammer_channels, dtype=float)
        # array和asarray都可以将结构数据转化为ndarray，但当数据源是ndarray时，array仍然会copy出一个副本，占用新的内存，asarray不会
        #self.jammer_indexs += self.step_forward
        print("jammer_channels", self.jammer_channels)
        # self.random_choose_channel()

        self.uavs = []  # 保存实体化的uav对象
        self.jammers = []  # 保存实体化的干扰机对象
        self.renew_uavs(self.n_uav)  # 初始化所有无人机的位置
        self.renew_jammers(self.n_jammer)  # 初始化所有干扰机的位置

        self.UAVchannels = UAVchannels(self.n_uav, self.n_channel, self.BS_position)
        self.Jammerchannels = Jammerchannels(self.n_jammer, self.n_uav, self.n_channel, self.BS_position)
        self.renew_channels()
        self.renew_uav_pairs()
        self.renew_neighbors_of_uavs()

    def get_state(self):
        if self.policy == "Q_learning":
            #uav_state = 0
            joint_state = np.zeros([self.n_transmitter], dtype=int)
            # perm，智能体可以观测到当前某个干扰机是干扰的哪一个信道，没有虚警漏警的可能
            # comb,智能体只能感知信道是否被干扰，不知道某个干扰机干扰的是哪个信道，有虚警漏警的可能
            if isinstance(self.jammer_channels, list):  # 扫频干扰，干扰信道观测是list
                jammer_channels = sorted(self.jammer_channels)   # 从小到大排序
                channels_observed = tuple(jammer_channels)     #list变成tuple
            else:    # 马尔可夫干扰，干扰信道观测是tuple
                channels_observed = self.jammer_channels
            # for i in range(self.n_transmitter):
            #     uav_state += self.uav_channels[i] * (self.action_range ** i)
            uav_state = self.uav_channels
            observed_state_idx = self.all_observed_states_list.index(channels_observed)
            #joint_state = uav_state * self.observed_state_dim + observed_state_idx
            for i in range(self.n_transmitter):
                joint_state[i] = uav_state[i] * self.observed_state_dim + observed_state_idx
            return joint_state

        elif self.policy == "drl":
            '''
            无人机的状态分为两部分：
            一是自己是否通信成功，1为成功，0为失败，该元素位于观察列表第一位
            二是对所有信道的监测，如果有人在信道上发送信息（无论无人机还是干扰机）设为1， 无人使用的信道设为0
            因此联合状态是一个n_transmitter x (1+n_channel)的二维列表
            '''
            joint_state = []
            for uav_id in range(self.n_transmitter):
                #channel_beifen = self.uav_channels
                #success_or_not = 0  # 通信成功与否的标志
                # 将干扰机干扰的信道打包为列表
                if not isinstance(self.jammer_channels, list):
                    jammer_channels = list(self.jammer_channels)
                else:
                    jammer_channels = self.jammer_channels
                # 将所有无人机选择的信道打包为列表
                if not isinstance(self.uav_channels, list):
                    uav_channels = deepcopy(list(self.uav_channels))
                else:
                    uav_channels = deepcopy(self.uav_channels)
                all_channels_chosed = list(set(jammer_channels + uav_channels))   # 统计所有被选择过的信道
                my_channel = uav_channels[uav_id]  # 提取当前无人机的信道
                del uav_channels[uav_id]  # 从列表中删除自己的信息
                # 判断是否通信成功
                if my_channel in uav_channels:  # 如果我选择的信道其他人也选了，通信失败
                    success_or_not = 0
                elif my_channel in jammer_channels:  # 如果我选择的信道干扰机选了，通信失败
                    success_or_not = 0
                else:  # 我选的信道干扰机没有选其他人也没有选，通信成功
                    success_or_not = 1
                # 给出当前无人机对信道的观察，如果信道
                if self.p_md == 0 and self.p_fa == 0:
                    channels_observed = np.zeros([self.n_channel], dtype=int)
                    channels_observed[all_channels_chosed] = 1
                else:
                    channels_observed = np.zeros([self.n_channel], dtype=int)
                    for i in range(self.n_channel):
                        if i in jammer_channels:
                            if random.random() < self.p_md:
                                channels_observed[i] = 0  # 漏警
                            else:
                                channels_observed[i] = 1  # 发现干扰
                        else:
                            if random.random() < self.p_fa:
                                channels_observed[i] = 1  # 虚警
                            else:
                                channels_observed[i] = 0  # 发现未干扰
                my_state = np.insert(channels_observed, obj=0, values=success_or_not)
                joint_state.append(my_state)
                #self.uav_channels = channel_beifen
            return joint_state

        else:  # 返回全局状态，是个列表，列表元素单个智能体观察的状态
            uav_channels = self.uav_channels
            # perm，智能体可以观测到当前某个干扰机是干扰的哪一个信道，没有虚警漏警的可能
            # comb,智能体只能感知信道是否被干扰，不知道某个干扰机干扰的是哪个信道，有虚警漏警的可能
            if isinstance(self.jammer_channels, list):  # 扫频干扰，干扰信道观测是list
                jammer_channels = sorted(self.jammer_channels)  # 从小到大排序
                channels_observed = tuple(jammer_channels)  # list变成tuple
            else:  # 马尔可夫干扰，干扰信道观测是tuple
                channels_observed = self.jammer_channels
            joint_state = np.concatenate((uav_channels, channels_observed))
            return joint_state

    def compute_reward(self, i, other_transmitter_channel_list, other_transmitter_index_list):
        channel = self.all_actions_for_each_transmitter[self.uav_channels[i]]

        uav_signal = np.zeros([self.n_des])
        uav_interference = np.zeros([self.n_des])   # 其他的transmitter对transmitter i的干扰
        uav_interference_from_jammer0 = np.zeros([self.n_des])    #后半段干扰机干扰
        uav_interference_from_jammer1 = np.zeros([self.n_des])    #前半段干扰机干扰
        uav_rate = np.zeros([self.n_des])
        uav_throughput = np.zeros([self.n_des])
        for j in range(self.n_des):
            transmitter_idx = self.uav_pairs[i][j][0]
            receiver_idx = self.uav_pairs[i][j][1]
            uav_signal[j] = 10 ** ((self.uav_power - self.UAVchannels_with_fastfading[transmitter_idx, receiver_idx, channel[j]] +
                                    2 * self.uavAntGain - self.uavNoiseFigure) / 10)
            if channel[j] in other_transmitter_channel_list:
                index = np.where(other_transmitter_channel_list == channel[j])
                for k in range(len(index)):
                    other_transmitter_index = other_transmitter_index_list[index[k][0]]
                    uav_interference[j] += 10 ** ((self.uav_power - self.UAVchannels_with_fastfading[other_transmitter_index, receiver_idx, channel[j]] +
                                                   2 * self.uavAntGain - self.uavNoiseFigure) / 10)

            if channel[j] in self.jammer_channels_list:
                idx = np.where(self.jammer_channels_list == channel[j])
                if (self.jammer_time[0] == self.t_Rx).all():     # 传输时间干扰机没换信道
                    for m in range(len(idx)):
                        jammer_idx = self.jammer_index_list[idx[m][0]]
                        uav_interference[j] += 10 ** ((self.jammer_power - self.Jammerchannels_with_fastfading[jammer_idx, receiver_idx, channel[j]] +
                                                       self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10)
                    uav_rate[j] = np.log2(1 + np.divide(uav_signal[j], (uav_interference[j] + self.sig2)))
                    if uav_rate[j] >= self.uav_rate:
                        uav_throughput[j] = self.uav_rate * self.bandwidth * self.jammer_time[0]
                    else:
                        uav_throughput[j] = 0
                else:    # 传输时间干扰机换了信道，判断干扰了前半段还是后半段
                    for m in range(len(idx)):
                        jammer_idx = self.jammer_index_list[idx[m][0]]
                        if idx[m][0] % 2 == 0:   # 后半段(self.jammer_channels_list先存入的后半段干扰信道序号）
                            uav_interference_from_jammer0[j] += 10 ** ((self.jammer_power - self.Jammerchannels_with_fastfading[jammer_idx, receiver_idx, channel[j]] +
                                                                      self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10)

                    uav_rate[j] = np.log2(1 + np.divide(uav_signal[j], (uav_interference[j] + uav_interference_from_jammer0[j] + self.sig2)))
                    if uav_rate[j] >= self.uav_rate:
                        uav_throughput[j] += self.uav_rate * self.bandwidth * self.jammer_time[0]
                    else:
                        uav_throughput[j] = 0

                    for l in range(len(idx)):
                        jammer_idx = self.jammer_index_list[idx[l][0]]
                        if idx[l][0] % 2 == 1:   # 前半段
                            uav_interference_from_jammer1[j] += 10 ** ((self.jammer_power - self.Jammerchannels_with_fastfading[jammer_idx, receiver_idx, channel[j]] +
                                                                      self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10)
                    uav_rate[j] = np.log2(1 + np.divide(uav_signal[j], (uav_interference[j] + uav_interference_from_jammer1[j] + self.sig2)))
                    if uav_rate[j] >= self.uav_rate:
                        uav_throughput[j] += self.uav_rate * self.bandwidth * self.jammer_time[1]
                    else:
                        uav_throughput[j] = 0

            else:
                uav_rate[j] = np.log2(1 + np.divide(uav_signal[j], (uav_interference[j] + self.sig2)))
                if uav_rate[j] >= self.uav_rate:
                    uav_throughput[j] = self.uav_rate * self.bandwidth * self.t_Rx
                else:
                    uav_throughput[j] = 0
        # print(uav_rate >= self.uav_rate)
        print(uav_rate, uav_signal, uav_interference)
        return uav_throughput

    def get_reward(self):
        uav_rewards = np.zeros([self.n_transmitter, self.n_des], dtype=float)
        self.uav_r = np.zeros([self.n_transmitter, self.n_des], dtype=float)
        self.uav_correct = np.zeros([self.n_transmitter, self.n_des], dtype=float)
        if self.jammer_channels_list == []:
            for i in range(self.n_jammer):
                self.jammer_channels_list.append(self.jammer_channels[i])
                self.jammer_index_list.append(i)
            self.jammer_time[0] = self.t_Rx
        print(self.jammer_channels_list)
        for i in range(self.n_transmitter):
            other_transmitter_channel_list = []
            other_transmitter_index_list = []
            for j in range(self.n_transmitter):
                if i==j:
                    continue
                else:
                    for k in range(self.n_des):
                        other_transmitter_channel_list.append(self.all_actions_for_each_transmitter[self.uav_channels[j]][k])
                        other_transmitter_index_list.append(self.uav_pairs[j][k][0])
            for m in range(self.n_des):
                uav_rewards[i][m] = self.compute_reward(i, other_transmitter_channel_list, other_transmitter_index_list)/self.max_throughput
                if uav_rewards[i][m]==self.uav_rate*self.t_Rx:
                    self.uav_correct[i][m] = 1
                channel = self.all_actions_for_each_transmitter[self.uav_channels[i]][m]
                if channel in self.jammer_channels_list or channel in other_transmitter_channel_list:
                    self.uav_r[i][m] = 0
                else:
                    self.uav_r[i][m] = 1
        #print(self.uav_r)
        self.jammer_channels_list = []
        self.jammer_index_list = []
        self.jammer_time = np.zeros([2])

        return uav_rewards

    def renew_jammer_channels_after_Rx(self):  # 更新干扰机的干扰信道（jammer_channels）和变换的timing（jammer_time）
        self.t_uav += self.t_Rx
        self.t_jammer += self.t_Rx
        if np.floor_divide((self.t_jammer - self.t_Rx), self.t_dwell) == np.floor_divide(self.t_jammer, self.t_dwell) - 1:
            if self.type_of_interference == "saopin":
                for i in range(self.n_jammer):
                    self.jammer_channels[i] += self.step_forward
                    self.jammer_channels[i] = int(self.jammer_channels[i] % self.n_channel)
                if self.t_jammer % self.t_dwell == 0:
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append((self.jammer_channels[i] + self.n_channel-1) % self.n_channel)
                        self.jammer_index_list.append(i)
                    self.jammer_time[0] = self.t_Rx
                else:  # 正好在Rx中间切换干扰信道
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append(self.jammer_channels[i])   # 后半段
                        self.jammer_index_list.append(i)
                        self.jammer_channels_list.append((self.jammer_channels[i] + self.n_channel-1) % self.n_channel)  # jammer_channels[i]-1
                        self.jammer_index_list.append(i)
                    change_times = np.floor_divide(self.t_jammer, self.t_dwell)
                    change_point = change_times * self.t_dwell
                    self.jammer_time[0] = self.t_jammer - change_point   # 0对应传输后半段的干扰时间
                    self.jammer_time[1] = self.t_Rx - self.jammer_time[0]
            elif self.type_of_interference == "markov":
                old_jammer_channels = self.jammer_channels
                # 所有无人机干扰信道进行转移
                for j in range(self.n_jammer):
                    idx = self.cur_observed_states_list[j].index(self.jammer_channels[j])
                    p = self.p_trans[idx]
                    self.jammer_channels[j] = random.choices(self.cur_observed_states_list[j], weights=p, k=1)[0]
                if self.t_jammer % self.t_dwell == 0:  # 传输完成后切换干扰信道
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append(old_jammer_channels[i])
                        self.jammer_index_list.append(i)
                    self.jammer_time[0] = self.t_Rx
                else:  # 传输中切换干扰信道
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append(self.jammer_channels[i])  # 后半段
                        self.jammer_index_list.append(i)
                        self.jammer_channels_list.append(old_jammer_channels[i])  # jammer_channels[i]-1
                        self.jammer_index_list.append(i)
                    change_times = np.floor_divide(self.t_jammer, self.t_dwell)
                    change_point = change_times * self.t_dwell
                    self.jammer_time[0] = self.t_jammer - change_point  # 0对应传输后半段的干扰时间
                    self.jammer_time[1] = self.t_Rx - self.jammer_time[0]
            print("change_channels", self.jammer_channels)

    def renew_jammer_channels_after_learn(self):
        self.t_uav += self.timestep
        self.t_jammer += self.timestep
        if np.floor_divide((self.t_jammer - self.timestep), self.t_dwell) == np.floor_divide(self.t_jammer, self.t_dwell) - 1:
            if self.type_of_interference == "saopin":
                for i in range(self.n_jammer):
                    self.jammer_channels[i] += self.step_forward
                    self.jammer_channels[i] = int(self.jammer_channels[i] % self.n_channel)
                    self.jammer_channels_list.append(self.jammer_channels[i])
                    self.jammer_index_list.append(i)
                self.jammer_time[0] = self.t_Rx
            elif self.type_of_interference == "markov":
                for j in range(self.n_jammer):
                    idx = self.cur_observed_states_list[j].index(self.jammer_channels[j])
                    p = self.p_trans[idx]
                    self.jammer_channels[j] = random.choices(self.cur_observed_states_list[j], weights=p, k=1)[0]
                # if self.t_jammer % self.t_dwell == 0:  传输开始前切换干扰信道
                for i in range(self.n_jammer):
                    self.jammer_channels_list.append(self.jammer_channels[i])
                    self.jammer_index_list.append(i)
                self.jammer_time[0] = self.t_Rx
            print("change_channels", self.jammer_channels)

    def renew_positions_of_transmitters(self):  # self.timestep = 0.01
        # ========================================================
        # This function update the position of each vehicle
        # ===========================================================
        for i in self.transmitter_list:
            delta_distance = self.uavs[i].velocity * self.timestep  # self.timestep = 0.01
            change_direction = False
            self.uavs[i].velocity = random.uniform(10, 20)    # 随机选速度
            # 随机选方向
            if self.uavs[i].direction == "forward":
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.uavs[i].position[1] <= self.left_lanes[j]) and \
                            ((self.uavs[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 1/3):
                            self.uavs[i].position = [self.uavs[i].position[0] - (delta_distance - (self.left_lanes[j] - self.uavs[i].position[1])),
                                                     self.left_lanes[j], self.height]
                            self.uavs[i].direction = "left"
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.uavs[i].position[1] <= self.right_lanes[j]) and \
                                ((self.uavs[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0, 1) < 0.5):
                                self.uavs[i].position = [self.uavs[i].position[0] + (delta_distance - (self.right_lanes[j] - self.uavs[i].position[1])),
                                                         self.right_lanes[j], self.height]
                                self.uavs[i].direction = "right"
                                change_direction = True
                                break
                if change_direction == False:
                    self.uavs[i].position[1] += delta_distance

            if (self.uavs[i].direction == "backward") and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.right_lanes)):
                    if (self.uavs[i].position[1] >= self.right_lanes[j]) and \
                            ((self.uavs[i].position[1] - delta_distance) <= self.right_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 1/3):
                            self.uavs[i].position = [self.uavs[i].position[0] + (delta_distance - (self.uavs[i].position[1] - self.right_lanes[j])),
                                                     self.right_lanes[j], self.height]
                            # print ('down with right', self.uavs[i].position)
                            self.uavs[i].direction = "right"
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.left_lanes)):
                        if (self.uavs[i].position[1] >= self.left_lanes[j]) and \
                                ((self.uavs[i].position[1] - delta_distance) <= self.left_lanes[j]):
                            if (random.uniform(0, 1) < 0.5):
                                self.uavs[i].position = [self.uavs[i].position[0] - (delta_distance - (self.uavs[i].position[1] - self.left_lanes[j])),
                                                         self.left_lanes[j], self.height]
                                # print ('down with left', self.uavs[i].position)
                                self.uavs[i].direction = "left"
                                change_direction = True
                                break
                if change_direction == False:
                    self.uavs[i].position[1] -= delta_distance

            if (self.uavs[i].direction == "right") and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.forward_lanes)):
                    if (self.uavs[i].position[0] <= self.forward_lanes[j]) and \
                            ((self.uavs[i].position[0] + delta_distance) >= self.forward_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 1/3):
                            self.uavs[i].position = [self.forward_lanes[j], self.uavs[i].position[1] + (delta_distance -
                                                     (self.forward_lanes[j] - self.uavs[i].position[0])), self.height]
                            change_direction = True
                            self.uavs[i].direction = "forward"
                            break
                if change_direction == False:
                    for j in range(len(self.backward_lanes)):
                        if (self.uavs[i].position[0] <= self.backward_lanes[j]) and \
                                ((self.uavs[i].position[0] + delta_distance) >= self.backward_lanes[j]):
                            if (random.uniform(0, 1) < 0.5):
                                self.uavs[i].position = [self.backward_lanes[j], self.uavs[i].position[1] - (delta_distance -
                                                        (self.backward_lanes[j] - self.uavs[i].position[0])), self.height]
                                change_direction = True
                                self.uavs[i].direction = "backward"
                                break
                if change_direction == False:
                    self.uavs[i].position[0] += delta_distance

            if (self.uavs[i].direction == "left") and (change_direction == False):
                for j in range(len(self.backward_lanes)):
                    if (self.uavs[i].position[0] >= self.backward_lanes[j]) and \
                            ((self.uavs[i].position[0] - delta_distance) <= self.backward_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 1/3):
                            self.uavs[i].position = [self.backward_lanes[j], self.uavs[i].position[1] - (delta_distance -
                                                    (self.uavs[i].position[0] - self.backward_lanes[j])), self.height]
                            change_direction = True
                            self.uavs[i].direction = "backward"
                            break
                if change_direction == False:
                    for j in range(len(self.forward_lanes)):
                        if (self.uavs[i].position[0] >= self.forward_lanes[j]) and \
                                ((self.uavs[i].position[0] - delta_distance) <= self.forward_lanes[j]):
                            if (random.uniform(0, 1) < 0.5):
                                self.uavs[i].position = [self.forward_lanes[j], self.uavs[i].position[1] + (delta_distance -
                                                        (self.uavs[i].position[0] - self.forward_lanes[j])), self.height]
                                change_direction = True
                                self.uavs[i].direction = "forward"
                                break
                if change_direction == False:
                    self.uavs[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.uavs[i].position[0] < 0) or (self.uavs[i].position[1] < 0) or \
                    (self.uavs[i].position[0] > self.length) or (self.uavs[i].position[1] > self.width):
                # delete
                # print ('delete ', self.position[i])
                if (self.uavs[i].direction == "forward"):
                    self.uavs[i].direction = 'right'
                    self.uavs[i].position = [self.uavs[i].position[0], self.right_lanes[-1], self.height]
                else:
                    if (self.uavs[i].direction == "backward"):
                        self.uavs[i].direction = 'left'
                        self.uavs[i].position = [self.uavs[i].position[0], self.left_lanes[0], self.height]
                    else:
                        if (self.uavs[i].direction == 'left'):
                            self.uavs[i].direction = "forward"
                            self.uavs[i].position = [self.forward_lanes[0], self.uavs[i].position[1], self.height]
                        else:
                            if (self.uavs[i].direction == 'right'):
                                self.uavs[i].direction = "backward"
                                self.uavs[i].position = [self.backward_lanes[-1], self.uavs[i].position[1], self.height]

    def renew_positions_of_receivers(self):
        for i in self.receiver_list:
            rec_pos = [self.uavs[i].position[0], self.uavs[i].position[1]]
            tra_id = self.uavs[i].destinations[0]
            tra_pos = [self.uavs[tra_id].position[0], self.uavs[tra_id].position[1]]
            while True:
                tra_xrange = [max(0, tra_pos[0] - self.max_distance), min(self.length, tra_pos[0] + self.max_distance)]
                rec_pos[0] = random.uniform(tra_xrange[0], tra_xrange[1])
                tra_yrange = [max(0, tra_pos[1] - self.max_distance), min(self.width, tra_pos[1] + self.max_distance)]
                rec_pos[1] = random.uniform(tra_yrange[0], tra_yrange[1])
                dis = (rec_pos[0] - tra_pos[0]) ** 2 + (rec_pos[1] - tra_pos[1]) ** 2
                if dis <= self.max_distance ** 2:
                    break

            self.uavs[i].position[0] = rec_pos[0]
            self.uavs[i].position[1] = rec_pos[1]

    def renew_positions_of_jammers(self):  # self.timestep = 0.01
        # ========================================================
        # This function update the position of each vehicle
        # ===========================================================
        i = 0
        # for i in range(len(self.position)):
        while (i < len(self.jammers)):
            # print ('start iteration ', i)
            # print(self.position, len(self.position), self.direction)
            delta_distance = self.jammers[i].velocity * self.timestep
            change_direction = False
            self.jammers[i].velocity = random.uniform(10, 20)   # 随机选速度
            # 随机选方向
            if self.jammers[i].direction == "forward":
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.jammers[i].position[1] <= self.left_lanes[j]) and \
                            ((self.jammers[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 1/3):
                            self.jammers[i].position = [self.jammers[i].position[0] - (delta_distance - (self.left_lanes[j] - self.jammers[i].position[1])),
                                                        self.left_lanes[j], self.height]
                            self.jammers[i].direction = "left"
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.jammers[i].position[1] <= self.right_lanes[j]) and \
                                ((self.jammers[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0, 1) < 0.5):
                                self.jammers[i].position = [self.jammers[i].position[0] + (delta_distance - (self.right_lanes[j] - self.jammers[i].position[1])),
                                                            self.right_lanes[j], self.height]
                                self.jammers[i].direction = "right"
                                change_direction = True
                                break
                if change_direction == False:
                    self.jammers[i].position[1] += delta_distance

            if (self.jammers[i].direction == "backward") and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.right_lanes)):
                    if (self.jammers[i].position[1] >= self.right_lanes[j]) and \
                            ((self.jammers[i].position[1] - delta_distance) <= self.right_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 1/3):
                            self.jammers[i].position = [self.jammers[i].position[0] + (delta_distance - (self.jammers[i].position[1] - self.right_lanes[j])),
                                                        self.right_lanes[j], self.height]
                            # print ('down with right', self.jammers[i].position)
                            self.jammers[i].direction = "right"
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.left_lanes)):
                        if (self.jammers[i].position[1] >= self.left_lanes[j]) and \
                                ((self.jammers[i].position[1] - delta_distance) <= self.left_lanes[j]):
                            if (random.uniform(0, 1) < 0.5):
                                self.jammers[i].position = [self.jammers[i].position[0] - (delta_distance - (self.jammers[i].position[1] - self.left_lanes[j])),
                                                            self.left_lanes[j], self.height]
                                # print ('down with left', self.jammers[i].position)
                                self.jammers[i].direction = "left"
                                change_direction = True
                                break
                if change_direction == False:
                    self.jammers[i].position[1] -= delta_distance

            if (self.jammers[i].direction == "right") and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.forward_lanes)):
                    if (self.jammers[i].position[0] <= self.forward_lanes[j]) and \
                            ((self.jammers[i].position[0] + delta_distance) >= self.forward_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 1/3):
                            self.jammers[i].position = [self.forward_lanes[j], self.jammers[i].position[1] + (delta_distance -
                                                        (self.forward_lanes[j] - self.jammers[i].position[0])), self.height]
                            change_direction = True
                            self.jammers[i].direction = "forward"
                            break
                if change_direction == False:
                    for j in range(len(self.backward_lanes)):
                        if (self.jammers[i].position[0] <= self.backward_lanes[j]) and \
                                ((self.jammers[i].position[0] + delta_distance) >= self.backward_lanes[j]):
                            if (random.uniform(0, 1) < 0.5):
                                self.jammers[i].position = [self.backward_lanes[j], self.jammers[i].position[1] - (delta_distance -
                                                           (self.backward_lanes[j] - self.jammers[i].position[0])), self.height]
                                change_direction = True
                                self.jammers[i].direction = "backward"
                                break
                if change_direction == False:
                    self.jammers[i].position[0] += delta_distance

            if (self.jammers[i].direction == "left") and (change_direction == False):
                for j in range(len(self.backward_lanes)):
                    if (self.jammers[i].position[0] >= self.backward_lanes[j]) and \
                            ((self.jammers[i].position[0] - delta_distance) <= self.backward_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 1/3):
                            self.jammers[i].position = [self.backward_lanes[j], self.jammers[i].position[1] - (delta_distance -
                                                       (self.jammers[i].position[0] - self.backward_lanes[j])), self.height, self.height]
                            change_direction = True
                            self.jammers[i].direction = "backward"
                            break
                if change_direction == False:
                    for j in range(len(self.forward_lanes)):
                        if (self.jammers[i].position[0] >= self.forward_lanes[j]) and \
                                ((self.jammers[i].position[0] - delta_distance) <= self.forward_lanes[j]):
                            if (random.uniform(0, 1) < 0.5):
                                self.jammers[i].position = [self.forward_lanes[j], self.jammers[i].position[1] + (delta_distance -
                                                           (self.jammers[i].position[0] - self.forward_lanes[j])), self.height]
                                change_direction = True
                                self.jammers[i].direction = "forward"
                                break
                if change_direction == False:
                    self.jammers[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.jammers[i].position[0] < 0) or (self.jammers[i].position[1] < 0) or \
                    (self.jammers[i].position[0] > self.length) or (self.jammers[i].position[1] > self.width):
                # delete
                # print ('delete ', self.position[i])
                if (self.jammers[i].direction == "forward"):
                    self.jammers[i].direction = 'right'
                    self.jammers[i].position = [self.jammers[i].position[0], self.right_lanes[-1], self.height]
                else:
                    if (self.jammers[i].direction == "backward"):
                        self.jammers[i].direction = 'left'
                        self.jammers[i].position = [self.jammers[i].position[0], self.left_lanes[0], self.height]
                    else:
                        if (self.jammers[i].direction == 'left'):
                            self.jammers[i].direction = "forward"
                            self.jammers[i].position = [self.forward_lanes[0], self.jammers[i].position[1], self.height]
                        else:
                            if (self.jammers[i].direction == 'right'):
                                self.jammers[i].direction = "backward"
                                self.jammers[i].position = [self.backward_lanes[-1], self.jammers[i].position[1], self.height]

            i += 1

    # def update_slow_fading(self):     # time slot is 2 ms.
    #     # ===========================================================================
    #     # This function updates all the channels including V2V and V2I channels
    #     # =============================================================================
    #     positions = [c.position for c in self.uavs]
    #     self.Jammerchannels.update_positions(positions)
    #     self.UAVchannels.update_positions(positions)
    #     self.Jammerchannels.update_pathloss()
    #     self.UAVchannels.update_pathloss()
    #     delta_distance = self.timeslot * np.asarray([c.velocity for c in self.uavs])  # time slot is 2 ms.
    #     self.Jammerchannels.update_shadow(delta_distance)
    #     self.UAVchannels.update_shadow(delta_distance)
    #     self.V2V_channels_abs = self.UAVchannels.PathLoss + self.UAVchannels.Shadow + 50 * np.identity(len(self.uavs))  # np.identity生成单位方阵???
    #     self.V2I_channels_abs = self.Jammerchannels.PathLoss + self.Jammerchannels.Shadow

    def renew_channels(self):
        # =======================================================================
        # This function updates all the channels including V2V and V2I channels
        # =========================================================================
        uav_positions = [u.position for u in self.uavs]
        jammer_positions = [j.position for j in self.jammers]
        self.Jammerchannels.update_positions(jammer_positions, uav_positions)
        self.UAVchannels.update_positions(uav_positions)
        self.Jammerchannels.update_pathloss()
        self.UAVchannels.update_pathloss()
        self.Jammerchannels.update_fast_fading()
        self.UAVchannels.update_fast_fading()
        UAVchannels_with_fastfading = np.repeat(self.UAVchannels.PathLoss[:, :, np.newaxis], self.n_channel, axis=2)
        self.UAVchannels_with_fastfading = UAVchannels_with_fastfading - self.UAVchannels.FastFading
        Jammerchannels_with_fastfading = np.repeat(self.Jammerchannels.PathLoss[:, :, np.newaxis], self.n_channel, axis=2)
        self.Jammerchannels_with_fastfading = Jammerchannels_with_fastfading - self.Jammerchannels.FastFading

    def act(self):
        self.renew_jammer_channels_after_Rx()
        reward = self.get_reward()
        self.renew_positions_of_transmitters()
        self.renew_positions_of_receivers()
        if self.is_jammer_moving:
            self.renew_positions_of_jammers()
        self.renew_channels()
        #self.uav_channels_real = action_real
        return reward

    def get_action(self, action):
        if self.policy == "Q_learning":
            action_all = np.zeros([self.n_transmitter, self.n_transmitter], dtype=int)
            #action_cur = np.zeros([self.n_uav], dtype=int)
            for i in range(self.n_transmitter):
                for j in range(self.n_transmitter - 1, -1, -1):
                    action_all[i, j] = int(np.floor(action[i] / (self.action_range**j)))
                    action[i] -= action_all[i, j] * (self.action_range**j)

            #self.uav_channels = action_all
            action_real = action_all.diagonal()
            print(action_real)
            return action_real
        elif self.policy == "drl":
            return action
        else:
            action_all = np.zeros([self.n_transmitter, self.n_transmitter], dtype=int)
            # action_cur = np.zeros([self.n_uav], dtype=int)
            for i in range(self.n_transmitter):
                for j in range(self.n_transmitter - 1, -1, -1):
                    action_all[i, j] = int(np.floor(action[i] / (self.action_range ** j)))
                    action[i] -= action_all[i, j] * (self.action_range ** j)

            # self.uav_channels = action_all
            action_real = action_all.diagonal()
            print(action_real)
            return action_real

    def reset(self):
        self.new_random_game()
        state = self.get_state()
        return state

    def step(self, a):
        #action = deepcopy(a)
        self.uav_channels = deepcopy(a)
        reward = self.act()
        state_next = self.get_state()  # 得到新的状态
        done = False  # 不会中途停止，一直走完整个回合

        return state_next, reward, done, {}