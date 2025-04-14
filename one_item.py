import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import scipy.io

from multiprocessing import Pool
import math
from scipy.io import savemat
from barweb import barweb

# 用于存储实验数据
import pickle

# 用于计算混合模型
#################################
## pass waiting for complement ##
#################################
import JV_extension as JV
#-----------CANN parameters--------
# neuron numbers
N = 100
# constants controlling the strength of the neuronal interaction J=8.3
J = 4.0
J0 = -0.9
B = 4.0
J_IE = 1.1
J_EI = 0.13
tau = 0.01
alpha = 1.5
I0 = -2.8
#----------STP parameters--------

tau_f = 6.1
tau_d = 0.36
U = 0.15

num_sub = 50
num_trial = 300
T_maintaining = np.concatenate((np.arange(0.1, 0.6, 0.1), [1, 2, 4, 6, 8, 10]))
#-----------Parameters of External Stimulus--------

#-----------Parameters of External Stimulus--------
A_loading_ext = 13.0
B_loading_ext = B

T_loading1 = 1.0
T_gap = 0.5
T_loading2 = 1.0

dt = 0.001


#------------Neurons-Related Variable Declaration----------
pos = np.linspace(-np.pi / 2, np.pi / 2, N + 1)[:-1]
exppos = np.exp(1j * pos * 2)
#-------------神经元之间的连接强度--------
J_amplitude = -J0 * np.ones((N, N))
diff_ang = np.mod(np.subtract.outer(pos, pos) + np.pi / 2, np.pi) - np.pi / 2
ind_interact_range = np.abs(B * diff_ang) <= np.arccos(J0 / J)
J_temp = J * np.cos(B * diff_ang)
J_amplitude[ind_interact_range] = J_temp[ind_interact_range]

sig1_recall = np.zeros((len(T_maintaining), num_sub))
sig2_recall = np.zeros((len(T_maintaining), num_sub))

#实验中记忆的角度、解码角度，以及 解码所用发放率的取值
all_theta_1 = np.zeros((len(T_maintaining), num_sub, num_trial))
all_theta_2 = np.zeros((len(T_maintaining), num_sub, num_trial))
all_theta_decode = np.zeros((len(T_maintaining), num_sub, num_trial))

# recall_perf 只用了两列
recall_perf = np.zeros((len(T_maintaining), 2))

recall1_perf_subj = np.zeros((len(T_maintaining), num_sub))
recall2_perf_subj = np.zeros((len(T_maintaining), num_sub))

######### mark #########
def simulate(m):
    theta_decode = np.zeros((num_sub, num_trial))
    T = T_loading1 + T_gap + T_loading2 + T_maintaining[m] + T_recalling
    N_t = int(T / dt)
    N_t_maintaining = int((T_loading1 + T_gap + T_loading2 + T_maintaining[m]) / dt)
    
    # 生成随机量
    theta1_tri = np.pi * np.random.rand(num_sub, num_trial) - np.pi / 2
    delta_theta_available = np.deg2rad([17, -17, 24, -24, 38, -38, 52, -52, 66, -66, 80, -80])
    delta_theta = delta_theta_available[np.random.randint(len(delta_theta_available), size=(num_sub, num_trial))]
    theta2_tri = np.mod(theta1_tri + delta_theta + np.pi / 2, np.pi) - np.pi / 2

    I_ext = np.zeros((N, N_t))
    u = np.zeros((N, N_t))
    x = np.ones((N, N_t))
    h_E = np.zeros((N, N_t))
    r_E = np.zeros((N, N_t))
    h_I = np.zeros(N_t)
    r_I = np.zeros(N_t)
    
    for subj in range(num_sub):
        for tri in range(num_trial):
            theta1 = theta1_tri[subj, tri]
            theta2 = theta2_tri[subj, tri]
            theta_recall = theta1 if tri <= num_trial / 2 else theta2
 
            # 编码期的外界刺激矩阵：defined by time steps
            diff = np.mod(pos - theta1 + np.pi / 2, np.pi) - np.pi / 2
            a_ext_loading1 = A_loading_ext * np.cos(B_loading_ext * diff) * (np.abs(B_loading_ext * diff) <= np.pi / 2)
            I_ext[:, :int(T_loading1 / dt)] = np.tile(a_ext_loading1, (int(T_loading1 / dt), 1)).T + sig_load * np.random.randn(N, int(T_loading1 / dt))

            diff = np.mod(pos - theta2 + np.pi / 2, np.pi) - np.pi / 2
            a_ext_loading2 = A_loading_ext * np.cos(B_loading_ext * diff) * (np.abs(B_loading_ext * diff) <= np.pi / 2)
            NT1 = int((T_loading1 + T_gap) / dt)
            NT2 = int((T_loading1 + T_gap + T_loading2) / dt)
            I_ext[:, NT1:NT2] = np.tile(a_ext_loading2, (NT2 - NT1, 1)).T + sig_load * np.random.randn(N, NT2 - NT1)

            

            for t in range(N_t_maintaining):
                du = (-u[:, t] / tau_f + U * (1 - u[:, t]) * r_E[:, t]) * dt
                u[:, t + 1] = u[:, t] + du

                dh_E = (-h_E[:, t] + J_amplitude @ (u[:, t + 1] * x[:, t] * r_E[:, t]) * np.pi / N - J_EI * r_I[t] + I_ext[:, t] + I0) * dt / tau
                h_E[:, t + 1] = h_E[:, t] + dh_E + np.sqrt(dt / tau) * sig1 * np.random.randn(N)
                r_E[:, t + 1] = alpha * np.log(1 + np.exp(h_E[:, t + 1] / alpha))

                dx = ((1 - x[:, t]) / tau_d - u[:, t + 1] * x[:, t] * r_E[:, t]) * dt
                x[:, t + 1] = x[:, t] + dx

                dh_I = (-h_I[t] + J_IE * np.sum(r_E[:, t]) * np.pi / N) * dt / tau
                h_I[t + 1] = h_I[t] + dh_I
                r_I[t + 1] = alpha * np.log(1 + np.exp(h_I[t + 1] / alpha))

            for t in range(N_t_maintaining, N_t - 1):
                du = (-u[:, t] / tau_f + U * (1 - u[:, t]) * r_E[:, t]) * dt
                u[:, t + 1] = u[:, t] + du

                dh_E = (-h_E[:, t] + J_amplitude @ (u[:, t + 1] * x[:, t] * r_E[:, t]) * np.pi / N - J_EI * r_I[t] + I_ext[:, t] + I0) * dt / tau
                h_E[:, t + 1] = h_E[:, t] + dh_E + np.sqrt(dt / tau) * (sig1 + sig2) * np.random.randn(N)
                r_E[:, t + 1] = alpha * np.log(1 + np.exp(h_E[:, t + 1] / alpha))

                dx = ((1 - x[:, t]) / tau_d - u[:, t + 1] * x[:, t] * r_E[:, t]) * dt
                x[:, t + 1] = x[:, t] + dx

                dh_I = (-h_I[t] + J_IE * np.sum(r_E[:, t]) * np.pi / N) * dt / tau
                h_I[t + 1] = h_I[t] + dh_I
                r_I[t + 1] = alpha * np.log(1 + np.exp(h_I[t + 1] / alpha))

            # 解码回忆期角度
            mr = np.mean(r_E[:, NT1:NT2], axis=1)
            theta_decode_tri = np.angle(np.sum(exppos * mr)) / 2
            theta_decode[subj, tri] = theta_decode_tri

            all_theta_decode[m, subj, tri] = theta_decode[subj, tri]

    tem1 = []
    tem2 = []

    for subj in range(num_sub):
        recall1_perf_subj_tem, LL1 = JV.JV10_fit(
        theta_decode[subj, :num_trial // 2].T * 2,
        theta1_tri[subj, :num_trial // 2].T * 2,
        theta2_tri[subj, :num_trial // 2].T * 2,
    )

        recall2_perf_subj_tem, LL2 = JV.JV10_fit(
        theta_decode[subj, num_trial // 2:].T * 2,
        theta2_tri[subj, num_trial // 2:].T * 2,
        theta1_tri[subj, num_trial // 2:].T * 2,
    )

        # 将结果存储在数组中
        recall1_perf_subj[m, subj] = recall1_perf_subj_tem[1]
        recall2_perf_subj[m, subj] = recall2_perf_subj_tem[1]
        tem1.append(recall1_perf_subj[m, subj])
        tem2.append(recall2_perf_subj[m, subj])

    recall1_perf = np.zeros(len(T_maintaining))
    recall2_perf = np.zeros(len(T_maintaining))

    sig1_recall[m, :] = np.array(tem1)
    sig2_recall[m, :] = np.array(tem2)

    # recall1_perf 和 recall2_perf 应该是一维数组
    recall1_perf[m] = np.mean(tem1)
    recall2_perf[m] = np.mean(tem2)

    all_theta_1[m, :, :] = theta1_tri
    all_theta_2[m, :, :] = theta2_tri 

    
if __name__ == '__main__':
    # 使用 8 个进程并行计算
    print('Start simulation')
    for i in range(len(T_maintaining)):
        simulate(i)
        print('T_maintaining =', T_maintaining[i], 'finished')

    print('sig1_recall:', sig1_recall)  # 检查 sig1_recall 是否生成
    print('sig2_recall:', sig2_recall)

    data_dict = {
    'sig1_recall': sig1_recall,
    'sig2_recall': sig2_recall,
    'all_theta_1': all_theta_1,
    'all_theta_2': all_theta_2,
    'all_theta_decode': all_theta_decode
    }

    # 保存为 .mat 文件
    savemat('sig1_recall_v2.mat', {'sig1_recall': sig1_recall})
    savemat('sig2_recall_v2.mat', {'sig2_recall': sig2_recall})
    savemat('all_theta_1_v2.mat', {'all_theta_1': all_theta_1})
    savemat('all_theta_2_v2.mat', {'all_theta_2': all_theta_2})
    savemat('all_theta_decode_v2.mat', {'all_theta_decode': all_theta_decode})


sig_theta1_temp = scipy.io.loadmat('sig1_recall_v2.mat')
sig_theta2_temp = scipy.io.loadmat('sig2_recall_v2.mat')

sig_recall1 = sig_theta1_temp['sig1_recall']
sig_recall2 = sig_theta2_temp['sig2_recall']

T_maintaining = np.concatenate((np.arange(0.1, 0.6, 0.1), [1], np.arange(2, 11, 2)))  # 记忆维持时间
num_T_maintaining = sig_recall1.shape[0]
tau_d = 0.4
num_sub = 50
comparision = []
std_com = []
std_recency = []

for m in range(len(T_maintaining)):
    sig1_recall1 = sig_recall1[m, :]
    sig2_recall2 = sig_recall2[m, :]
    
    mean_perf1 = np.mean(sig1_recall1)
    mean_perf2 = np.mean(sig2_recall2)
    std_perf1 = np.std(sig1_recall1)
    std_perf2 = np.std(sig2_recall2)
    
    std_recency_tem = np.std(sig1_recall1 - sig2_recall2)
    std_recency.append(std_recency_tem)
    
    comparision.append([mean_perf1, mean_perf2])
    std_com.append([std_perf1, std_perf2])
    
    # t-test 进行显著性检验
    h, p = ttest_ind(sig1_recall1, sig2_recall2)
    
    # 这里可以添加文本注释，但为了保持与原MATLAB代码一致，这里注释掉
    # if p < 0.001:
    #     plt.text(1.3, y_pos + 0.9, '***', fontsize=10)
    # elif p < 0.01:
    #     plt.text(1.3, y_pos + 0.9, '**', fontsize=10)
    # elif p < 0.05:
    #     plt.text(1.4, y_pos + 1, '*', fontsize=10)
    # else:
    #     plt.text(1.4, y_pos + 1.3, 'n.s.', fontsize=10)

comparision = np.array(comparision)
std_com = np.array(std_com)
std_recency = np.array(std_recency)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 绘制 Memory Performance 图
axs[0].bar(np.arange(len(T_maintaining)), np.arcsin(comparision[:, 0]), yerr=np.arcsin(std_com[:, 0]), width=0.4, label='item_1')
axs[0].bar(np.arange(len(T_maintaining)) + 0.4, np.arcsin(comparision[:, 1]), yerr=np.arcsin(std_com[:, 1]), width=0.4, label='item_2')
axs[0].set_xlabel('T_{maintaining} [s]', fontsize=11)
axs[0].set_ylabel('Memory Performance', fontsize=11)
axs[0].legend()
axs[0].set_ylim([1.1, 1.5])

# 绘制 Memory Bias 图
axs[1].bar(np.arange(len(T_maintaining)), np.arcsin(comparision[:, 1] - comparision[:, 0]), yerr=np.arcsin(std_recency) / (num_sub ** 0.5), width=0.4)
axs[1].set_xlabel('T_{maintaining} [s]', fontsize=11)
axs[1].set_ylabel('Memory Bias', fontsize=11)

plt.tight_layout()
plt.savefig('two_items.pdf')
plt.show()
