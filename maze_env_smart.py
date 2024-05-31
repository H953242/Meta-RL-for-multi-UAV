   # -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:35:13 2020

@author: Dora
"""

import numpy as np
import math

class Maze(object):

    def __init__(self):
        super(Maze, self).__init__()
        self.N = 3
        self.n_block =5
        self.n_power = 4
        self.power = np.zeros(self.N)
        self.b = 180*10**3 #带宽
        self.p_min = 0.01
        self.p_max = 0.15
        self.block = np.zeros(self.N)
        self.n_jam = 1
        self.jam_block = np.zeros(self.n_jam)
        self.jam_block[0] = 2
#        self.jam_block[1] = 3
        self.n_actions = 5
        self.n_state = [5*2**3,5*2**3,5*2**3,7*2**5,7*2**5]
        self.n_state_jam = [5**3,5**3]
                # self.n_state = [2**5,2**4,2**4,2**4,2**4]
#        self.Distance_UE = np.array([[0,20,20,20,20],[20,0,20*math.sqrt(2),20*math.sqrt(2),40],[20,20*math.sqrt(2),0,40,20*math.sqrt(2)],[20,20*math.sqrt(2),40,0,20*math.sqrt(2)],[20,40,20*math.sqrt(2),20*math.sqrt(2),0]])
        # self.TUE_x = np.array([0,-10*math.sqrt(2),10*math.sqrt(2),-10*math.sqrt(2),10*math.sqrt(2)])
        # self.TUE_y = np.array([0,10*math.sqrt(2),10*math.sqrt(2),-10*math.sqrt(2),-10*math.sqrt(2)])
        # self.RUE_x = np.array([0,-10*math.sqrt(2),10*math.sqrt(2),-10*math.sqrt(2),10*math.sqrt(2)])+1
        # self.RUE_y = np.array([0,10*math.sqrt(2),10*math.sqrt(2),-10*math.sqrt(2),-10*math.sqrt(2)])-1
        # self.TUE_x = np.array([0,-10*math.sqrt(2),10*math.sqrt(2),-10*math.sqrt(2)])
        # self.TUE_y = np.array([0,10*math.sqrt(2),10*math.sqrt(2),-10*math.sqrt(2)])
        # self.RUE_x = np.array([0,-10*math.sqrt(2),10*math.sqrt(2),-10*math.sqrt(2)])+1
        # self.RUE_y = np.array([0,10*math.sqrt(2),10*math.sqrt(2),-10*math.sqrt(2)])-1
#        self.Distance_UE = np.array([[0,20,20,20,20],[20,0,20*math.sqrt(2),20*math.sqrt(2),40],[20,20*math.sqrt(2),0,40,20*math.sqrt(2)],[20,20*math.sqrt(2),40,0,20*math.sqrt(2)],[20,40,20*math.sqrt(2),20*math.sqrt(2),0]])
        self.TUE_x = np.array([0,-10*math.sqrt(2),10*math.sqrt(2)])
        self.TUE_y = np.array([0,10*math.sqrt(2),10*math.sqrt(2)])
        self.RUE_x = np.array([0,-10*math.sqrt(2),10*math.sqrt(2)])+1
        self.RUE_y = np.array([0,10*math.sqrt(2),10*math.sqrt(2)])-1

        self.Jam_x = [-10]
        self.Jam_y = [0]
        self.Distance_TUEtoTUE = np.zeros((self.N,self.N))
        self.Distance_TUEtoRUE = np.zeros((self.N,self.N))
        self.Distance_JamtoRUE = np.zeros((self.n_jam,self.N))
        self.dis = 30 
        
        self.Noise = 10**(-174/10)*10**(-3)
        
        self.A = 0.2
        self.B = 0.1
        self.C = 2.0
        self.D = 0
        
        self.rate = np.zeros(self.N)
        self.rate_ref = [8*10**6,8*10**6,8*10**6,8*10**6,8*10**6]
        
#        self.Distance_jam_UE = np.array([[10,math.sqrt(500-200*math.sqrt(2)),math.sqrt(500+200*math.sqrt(2)),math.sqrt(500-200*math.sqrt(2)),math.sqrt(500+200*math.sqrt(2))],[10,math.sqrt(500-200*math.sqrt(2)),math.sqrt(500-200*math.sqrt(2)),math.sqrt(500+200*math.sqrt(2)),math.sqrt(500+200*math.sqrt(2))]])
        self.p_jam = 0.2
        self.QoS = np.zeros(self.N)
            
    def step(self, observation, action,jam_obs0,action_jam):
        sum_success = 0
        reward = np.zeros(self.N)
        reward0 = np.zeros(self.N)
        reward_jam = np.zeros(self.n_jam)
        jam_obs0_ = np.zeros(self.n_jam)
        QoE = np.ones(self.N)
        observation_ = np.zeros(self.N)
        jam_obs_ = np.zeros(self.n_jam)
        self.success = np.zeros(self.N)
        n_neighbor = [[] for i in range(self.N)]
        jam_observation = np.zeros((self.N,self.n_jam))
        # jam_p = [[0,0.5,0,0,0,0,0,0,0.5],[0.5,0,0.5,0,0,0,0,0,0],[0,0.5,0,0.5,0,0,0,0,0],[0,0,0.5,0,0.5,0,0,0,0],[0,0,0,0.5,0,0.5,0,0,0],[0,0,0,0,0.5,0,0.5,0,0],[0,0,0,0,0,0.5,0,0.5,0],[0,0,0,0,0,0,0.5,0,0.5],[0.5,0,0,0,0,0,0,0.5,0]]
        # jam_p = [[0,0.5,0,0.5],[0.5,0,0.5,0],[0,0.5,0,0.5],[0.5,0,0.5,0]]
        # jam_p = [[0,0.5,0,0,0.5],[0.5,0,0.5,0,0],[0,0.5,0,0.5,0],[0,0,0.5,0,0.5],[0.5,0,0,0.5,0]]
        # jam_p = [[0,0.5,0,0,0,0.5],[0.5,0,0.5,0,0,0],[0,0.5,0,0.5,0,0],[0,0,0.5,0,0.5,0],[0,0,0,0.5,0,0.5],[0.5,0,0,0,0.5,0]]
        # jam_p = [[0,0.5,0,0,0,0,0.5],[0.5,0,0.5,0,0,0,0],[0,0.5,0,0.5,0,0,0],[0,0,0.5,0,0.5,0,0],[0,0,0,0.5,0,0.5,0],[0,0,0,0,.5,0,0.5],[0.5,0,0,0,0,0.5,0]]
        # jam_p = [[0,0.5,0,0,0,0,0,0.5],[0.5,0,0.5,0,0,0,0,0],[0,0.5,0,0.5,0,0,0,0],[0,0,0.5,0,0.5,0,0,0],[0,0,0,0.5,0,0.5,0,0],[0,0,0,0,0.5,0,0.5,0],[0,0,0,0,0,0.5,0,0.5],[0.5,0,0,0,0,0,0.5,0]]
        # jam_p = [[0,0.5,0,0,0,0,0,0,0,0.5],[0.5,0,0.5,0,0,0,0,0,0,0],[0,0.5,0,0.5,0,0,0,0,0,0],[0,0,0.5,0,0.5,0,0,0,0,0],[0,0,0,0.5,0,0.5,0,0,0,0],[0,0,0,0,0.5,0,0.5,0,0,0],[0,0,0,0,0,0.5,0,0.5,0,0],[0,0,0,0,0,0,0.5,0,0.5,0],[0,0,0,0,0,0,0,0.5,0,0.5],[0.5,0,0,0,0,0,0,0,0.5,0]]
        interference = np.zeros(self.N)
        
        # 用户移动
        for i in range(self.N):
            
            if -20<=self.TUE_x[i]<=20:
                a = np.random.randint(0,2,size=1)
#                print('a',a)
                self.TUE_x[i] = a*(self.TUE_x[i]-0.5)+(1-a)*(self.TUE_x[i]+0.7)
                self.RUE_x[i] = a*(self.RUE_x[i]-0.5)+(1-a)*(self.RUE_x[i]+0.7)
            elif self.TUE_x[i]<-20:
                if -20<=self.TUE_y[i]<=20:
                    a = np.random.randint(0,2,size=1)
                    self.TUE_y[i] = a*(self.TUE_y[i]+0.3)+(1-a)*(self.TUE_y[i]-0.7)
                    self.TUE_x[i] +=0.6
                    self.RUE_y[i] = a*(self.RUE_y[i]+0.3)+(1-a)*(self.RUE_y[i]-0.7)
                    self.RUE_x[i] +=0.6
                else:
                    self.TUE_y[i] = self.TUE_y[i]-0.5
                    self.RUE_y[i] = self.RUE_y[i]-0.5
            else:
                if -20<=self.TUE_y[i]<=20:
                    a = np.random.randint(0,2,size=1)
                    self.TUE_y[i] = a*(self.TUE_y[i]+0.3)+(1-a)*(self.TUE_y[i]-0.7)
                    self.TUE_x[i] -=0.7
                    self.RUE_y[i] = a*(self.RUE_y[i]+0.3)+(1-a)*(self.RUE_y[i]-0.7)
                    self.RUE_x[i] -=0.7
                else:
                    self.TUE_x[i] -=0.8
                    self.RUE_x[i] -=0.8
           
        # 检测无人机间距离
        for i in range(self.N):
            for j in range(self.N):
                self.Distance_TUEtoTUE[i,j] =  math.sqrt((self.TUE_x[i]-self.TUE_x[j])**2+(self.TUE_y[i]-self.TUE_y[j])**2)
                self.Distance_TUEtoRUE[i,j] =  math.sqrt((self.TUE_x[i]-self.RUE_x[j])**2+(self.TUE_y[i]-self.RUE_y[j])**2)
            for k in range(self.n_jam):
                self.Distance_JamtoRUE[k,i] =  math.sqrt((self.RUE_x[i]-self.Jam_x[k])**2+(self.RUE_y[i]-self.Jam_y[k])**2)
            self.Distance_TUEtoTUE[i,i] =  math.sqrt((self.TUE_x[i]-self.RUE_x[i])**2+(self.TUE_y[i]-self.RUE_y[i])**2)
#        print('Distance_UE',self.Distance_UE)
        
        for i in range(self.n_jam):

            self.jam_block[i] = action_jam[i]
        
        print('jam_block',self.jam_block)
        
        self.gain = np.random.exponential(1,(self.N)).clip(0,1)
        self.gain_UE = np.random.exponential(1,(self.N,self.N)).clip(0,1)
        self.gain_jam = np.random.exponential(1,(self.N,self.n_jam)).clip(0,1)
        
        
        for i in range(self.N):         
            self.block[i] = action[i,0]           
            self.power[i] = self.p_min + action[i,1]/self.n_power*(self.p_max-self.p_min)
        print('block',self.block)
        print('power',self.power)

        for i in range(self.N):
            for j in range(self.N):
                if self.Distance_TUEtoTUE[i,j]<=self.dis:
                    n_neighbor[i].append(j)
        # print('n_neighbor',n_neighbor)

        for i in range(self.N):
            for j in range(len(n_neighbor[i])):
                if self.block[i]==self.block[j] and i!=j:
                    interference[i] += self.power[j]*self.Distance_TUEtoRUE[j,i]**(-2)*self.gain_UE[i,j]
                    QoE[i] = -1
            for j in range(self.n_jam):
                if self.block[i] == self.jam_block[j]:
                    interference[i] += self.p_jam*self.Distance_JamtoRUE[j,i]**(-2)*self.gain_jam[i,j]
#                    QoE[i] = -1
                    QoE[i] = 0
                    reward_jam[j] = 1
            self.rate[i] = self.b*math.log(1+self.power[i]*self.Distance_TUEtoTUE[i,i]**(-2)*self.gain[i]/(self.Noise+interference[i]),2)
        print('rate',self.rate)
        print('QoE',QoE)
        
#        success= np.zeros(self.N)
        for i in range(self.N):
            if self.rate[i]>=self.rate_ref[i]:
                self.success[i]=1
                self.QoS[i] = self.D + 1/(self.A+self.B*math.exp(-self.C*((self.rate[i]-self.rate_ref[i])/10**5)))-10*self.power[i]
            else:
                self.QoS[i] = self.D + 1/(self.A+self.B*math.exp(-self.C*((self.rate[i]-self.rate_ref[i])/10**5)))-5-10*self.power[i]
        print('QoS',self.QoS)
#        for i in range(self.N):
#            for j in range(self.N):
#                if self.block[i]==self.block[j] and i!=j and self.Distance_UE[i,j]<=self.dis:
#                    self.success[i] = 0  
#                    QoE[i] = -1
            # if self.gain[i,int(self.block[i])]<=0.1:
            # if self.gain[i,int(self.block[i])]==0:  
            #     self.success[i] = 0
#            for j in range(self.n_jam):
#                if self.block[i]==self.jam_block[j]:
#                    self.success[i] = 0
#                    QoE[i] =-1
              
        print('success',self.success)
      
        # for i in range(self.N):
        #     for j in range(self.N):
        #         if self.Distance_UE[i,j]<=self.dis:
        #             for e in range(self.n_jam):
        #                 if self.block[j] == self.jam_block[e]:
        #                     jam_observation[i,e] = self.jam_block[e]
        for i in range(self.N):
            for e in range(self.n_jam):
                jam_observation[i,e] = self.jam_block[e]                    
                jam_obs0_[e] += self.block[i]*5**i
        # print('jam_observation',jam_observation)

#        for i in range(self.n_jam):
#            for j in range(self.N):
#                if self.block[j] == self.jam_block[i]:
#                      jam_obs_[i] = self.jam_block[i]  
        jam_obs_=self.jam_block
        
#        print('n_neighbor',n_neighbor)
        for i in range(self.N):
            for j in range(len(n_neighbor[i])):
                observation_[i] += self.success[n_neighbor[i][j]]*2**n_neighbor[i][j]
                reward[i]+=self.QoS[n_neighbor[i][j]]
                reward0[i]+=QoE[n_neighbor[i][j]]
            observation_[i] += jam_observation[i,0]*2**self.N
            sum_success+=self.success[i]
           
        print('reward',reward)       
                    
        # print('observation',observation_)

        return observation_, reward,reward0, jam_obs_,sum_success,jam_obs0_,reward_jam

    def reset(self):
        self.N = 3      
        self.n_block = 5
        self.block = np.zeros(self.N)
        self.p = 0.1
        self.n_actions = 5
        observation = np.ones(self.N)
        jam_obs = np.zeros(self.n_jam)
        jam_obs[0]=2
        jam_obs0 =np.zeros(self.n_jam)+0*25+1*5+1  #[0,1,1]
#        jam_obs[1]=3
        return observation,jam_obs,jam_obs0






















