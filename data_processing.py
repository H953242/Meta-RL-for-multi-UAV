from pylab import *
import os
from common.arguments import get_args
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams['font.sans-serif'] = ['SimHei']

# 获取保存路径
args = get_args()
save_path = args.save_dir + '/' + args.scenario_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

mfq_softold = np.loadtxt('2021.11.05_mfq_500e_2000s')
mfq_soft001 = np.loadtxt('mfq_soft001_500e_2000s')
mfq_soft0003 = np.loadtxt('2021.11.16_mfq_500e_2000s')
mfq_greedy = np.loadtxt('mfq_final_notsoft_500e_2000s')
mfq_cs = np.load('mfq_soft001_10u13c_500e_2000s_channel_selected.npy')

uav0 = mfq_cs[0]
print(uav0)

x = np.array(range(args.time_steps))
y = np.array(range(args.episode))
x, y = np.meshgrid(x, y)
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(y, x, uav0, rstride=1, cstride=1, cmap='rainbow')
plt.show()
#plt.plot(x, mfq_soft001.sum(axis=0), ls='-', color='red', label='soft0.01')
#plt.plot(x, mfq_greedy.sum(axis=0), ls='-', color='blue', label='greedy')

plt.legend()
plt.xlabel("episode")
plt.ylabel("mean reward for time_steps times in one episode")
plt.title('无人机的单回合平均奖励变化')
# 保存图片到本地
plt.savefig(save_path + '/uav_channel.png', format='png')
plt.show()

