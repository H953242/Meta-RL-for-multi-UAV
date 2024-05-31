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
        self.N = 5
        self.n_block = 5
        self.block = np.zeros(self.N)    
        self.n_actions = 5
        self.n_state = [36*2**5,36*2**4,36*2**4,36*2**4,36*2**4]
        # self.n_state = [2**5,2**4,2**4,2**4,2**4]
        self.Distance_UE = np.array([[0,20,20,20,20],[20,0,20*math.sqrt(2),20*math.sqrt(2),40],[20,20*math.sqrt(2),0,40,20*math.sqrt(2)],[20,20*math.sqrt(2),40,0,20*math.sqrt(2)],[20,40,20*math.sqrt(2),20*math.sqrt(2),0]])
        self.dis = 30 
        self.p = 0.1
        self.Distance_jam_UE = np.array([[10,math.sqrt(500-200*math.sqrt(2)),math.sqrt(500+200*math.sqrt(2)),math.sqrt(500-200*math.sqrt(2)),math.sqrt(500+200*math.sqrt(2))],[10,math.sqrt(500-200*math.sqrt(2)),math.sqrt(500-200*math.sqrt(2)),math.sqrt(500+200*math.sqrt(2)),math.sqrt(500+200*math.sqrt(2))]])
        self.p_jam = 0.2
        self.n_jam = 2
        self.jam_block = np.zeros(self.n_jam)
        self.jam_block[0] = 2
        self.jam_block[1] = 3
            
    def step(self, observation, action):
        sum_success = 0
        reward = np.zeros(self.N)
        QoE = np.ones(self.N)
        observation_ = np.zeros(self.N)
        jam_obs_ = np.zeros(self.n_jam)
        self.success = np.ones(self.N)
        n_neighbor = [[] for i in range(self.N)]
        jam_observation = np.zeros((self.N,self.n_jam))+self.n_block
        jam_p = [[0,0.5,0,0,0.5],[0.5,0,0.5,0,0],[0,0.5,0,0.5,0],[0,0,0.5,0,0.5],[0.5,0,0,0.5,0]]
        
        for i in range(self.n_jam):
            # self.jam_block[i]+=1
            # if self.jam_block[i]==5:
            #     self.jam_block[i]=0
            a = int(self.jam_block[i])
            self.jam_block[i] = np.random.choice(5,size=1,p=jam_p[a])
        
        print('jam_block',self.jam_block)
        # self.gain = np.random.exponential(1,(self.N,self.n_block))
        # for i in range(self.N):
        #     self.gain[i,i] = np.random.exponential(2)
        # # self.gain = np.random.randint(2,size=(self.N,self.n_block))
        # print('gain',self.gain)
        for i in range(self.N):         
            self.block[i] = action[i]           
        print('block',self.block)

        for i in range(self.N):
            for j in range(self.N):
                if self.Distance_UE[i,j]<=self.dis:
                    n_neighbor[i].append(j)
        # print('n_neighbor',n_neighbor)


        for i in range(self.N):
            for j in range(self.N):
                if self.block[i]==self.block[j] and i!=j and self.Distance_UE[i,j]<=self.dis:
                    self.success[i] = 0  
                    QoE[i] = -1
            # if self.gain[i,int(self.block[i])]<=0.1:
            # if self.gain[i,int(self.block[i])]==0:  
            #     self.success[i] = 0
            for j in range(self.n_jam):
                if self.block[i]==self.jam_block[j]:
                    self.success[i] = 0
                    QoE[i] =-1
              
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
        
        # print('jam_observation',jam_observation)

        for i in range(self.n_jam):
            for j in range(self.N):
                if self.block[j] == self.jam_block[i]:
                      jam_obs_[i] = self.jam_block[i]  
        jam_obs_=self.jam_block
        
        # print('n_neighbor',n_neighbor)
        for i in range(self.N):
            for j in range(len(n_neighbor[i])):
                observation_[i] += self.success[n_neighbor[i][j]]*2**j
                reward[i]+=QoE[n_neighbor[i][j]]
            observation_[i] += jam_observation[i,0]*(self.n_block+1)*2**(len(n_neighbor[i]))+jam_observation[i,1]*2**(len(n_neighbor[i]))
            sum_success+=self.success[i]
        print('reward',reward)       
                    
        print('observation',observation_)        

        return observation_, reward, jam_obs_,sum_success

    def reset(self):
        self.N = 5      
        self.n_block = 5
        self.block = np.zeros(self.N)
        self.p = 0.1
        self.n_actions = 5
        observation = np.ones(self.N)
        jam_obs = np.zeros(self.n_jam)
        jam_obs[0]=2
        jam_obs[1]=3
        return observation,jam_obs






















