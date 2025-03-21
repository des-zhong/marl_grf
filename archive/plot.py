import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d
# Initialize an empty list to store values
# cur_eprewmean_values = np.array([[-1.15, -0.535, -0.694, -0.547, -0.228,  0.509],
#                             [0.812, 0.941,   1.16,   2.05,   1.94,    3.53],
#                             [1.47,   1.76,    2.05,   2.71,  2.59,    1.92],
#                             [1.24, 0.976, 1.35, 1.46, 0, 0]])
# baseline_eprewmean_values = np.array([-2.74, -1.93, -1.68, -1.75, -1.64, -1.67, -1.82, -1.22, -0.535, -0.694, -0.165, -0.365, -0.412, -0.453, 0.506, 0.165, 0.853, 0.394, 0.229])
cur_eprewmean_values = []
cur_ppo_mean_values = []
s=10
plt.figure(figsize=(10, 3.7))
plt.text(750, 1, "Phase 1", fontsize=12, color='r', ha='center', va='bottom')
plt.text(1750, 3.3, "Phase 2", fontsize=12, color='r', ha='center', va='bottom')
plt.text(3000, 3, "Phase 3", fontsize=12, color='r', ha='center', va='bottom')
plt.text(3800, 2, "Phase 4", fontsize=12, color='r', ha='center', va='bottom')
index = 0
# ours_path = ['2025-03-05_16-46-24','2025-03-06_01-25-43','2025-03-06_09-54-27','2025-03-06_18-22-34','2025-03-07_10-59-25']
ours_path = ['2025-03-05_16-46-24','2025-03-06_01-25-43','2025-03-06_09-54-27','2025-03-07_10-59-25']
cur_ppo_path = ['2025-03-06_14-04-19', '2025-03-06_22-10-29', '2025-03-07_06-34-29','2025-03-07_15-04-48']
for i in range(len(ours_path)):
    with open('ours/'+ours_path[i]+'/evg_return.pkl', 'rb') as f:
        a = np.array(pickle.load(f))
    index+=len(a)
    b = gaussian_filter1d(a, sigma=s)
    cur_eprewmean_values.append(b)
    if i<len(ours_path)-1:
        plt.axvline(x=index-45*i-45, color='k', linestyle='--', linewidth=2)

cur_eprewmean_values = np.concatenate(cur_eprewmean_values)
cur_eprewmean_values = cur_eprewmean_values[~np.isnan(cur_eprewmean_values)]




baseline_eprewmean_values = []
impala_path = ['2025-03-05_16-47-22', '2025-03-07_11-01-30']
for i in range(len(impala_path)):
    with open('impala/'+impala_path[i]+'/evg_return.pkl', 'rb') as f:
        a = np.array(pickle.load(f))
    b = gaussian_filter1d(a, sigma=s)
    baseline_eprewmean_values.append(b)
baseline_eprewmean_values[0]=baseline_eprewmean_values[0][:-len(baseline_eprewmean_values[1])]
baseline_eprewmean_values = np.concatenate(baseline_eprewmean_values)
baseline_eprewmean_values = baseline_eprewmean_values[~np.isnan(baseline_eprewmean_values)]
mlp_paths = ['2025-03-05_16-48-04','2025-03-07_11-02-44']
mlp_eprewmean_values = []
for i in range(len(mlp_paths)):
    with open('ppo/'+mlp_paths[i]+'/evg_return.pkl', 'rb') as f:
        a = np.array(pickle.load(f))
    b = gaussian_filter1d(a, sigma=s)
    mlp_eprewmean_values.append(b)
mlp_eprewmean_values[0]=mlp_eprewmean_values[0][:-len(mlp_eprewmean_values[1])]
mlp_eprewmean_values = np.concatenate(mlp_eprewmean_values)

mlp_eprewmean_values = mlp_eprewmean_values[~np.isnan(mlp_eprewmean_values)]



for i in range(len(cur_ppo_path)):
    with open('cur_ppo/'+cur_ppo_path[i]+'/evg_return.pkl', 'rb') as f:
        a = np.array(pickle.load(f))
    index+=len(a)
    b = gaussian_filter1d(a, sigma=s)
    cur_ppo_mean_values.append(b)

cur_ppo_mean_values = np.concatenate(cur_ppo_mean_values)

cur_ppo_mean_values = cur_ppo_mean_values[~np.isnan(cur_ppo_mean_values)]


plt.plot(range(len(cur_eprewmean_values)),cur_eprewmean_values, label='Ours')
plt.plot(range(len(cur_ppo_mean_values)),cur_ppo_mean_values, label='MAPPO+CR')
plt.plot(range(len(baseline_eprewmean_values)), baseline_eprewmean_values, label='IMPALA')
plt.plot(range(len(mlp_eprewmean_values)),mlp_eprewmean_values, label='MAPPO')
print('mean value: ours, ppo+cr, impala, ppo:')
print(np.mean(cur_eprewmean_values[-100:]), np.mean(cur_ppo_mean_values[-100:]), np.mean(baseline_eprewmean_values[-100:]), np.mean(mlp_eprewmean_values[-100:]))
# plt.plot(range(len(baseline_eprewmean_values)),np.ones(len(baseline_eprewmean_values)))
plt.xlabel("Iterations")
plt.ylabel("Episode Mean Return")
# plt.title("Episod Mean Return")
plt.legend()
plt.grid(True)
plt.savefig("compare.svg",format='svg')