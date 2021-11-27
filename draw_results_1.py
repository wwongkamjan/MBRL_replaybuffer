import csv
import numpy as np
import matplotlib.pyplot as plt
import  tikzplotlib

def bootstrapping(data, num_per_group, num_of_group):
    new_data = np.array([np.mean(np.random.choice(data, num_per_group, replace=True)) for _ in range(num_of_group)])
    return new_data

def generate_confidence_interval(ys, number_per_g = 30, number_of_g = 1000, low_percentile = 1, high_percentile = 99):
    means = []
    mins =[]
    maxs = []
    for i,y in enumerate(ys.T):
        y = bootstrapping(y, number_per_g, number_of_g)
        means.append(np.mean(y))
        mins.append(np.percentile(y, low_percentile))
        maxs.append(np.percentile(y, high_percentile))
    return np.array(means), np.array(mins), np.array(maxs)

def plot_ci(x, y, num_runs, num_dots, mylegend, ls='-', lw=1.5, transparency=0.25, color='red'):
    assert(x.ndim==1)
    assert(x.size==num_dots)
    assert(y.ndim==2)
    print(y.shape)
    assert(y.shape==(num_runs,num_dots))
    y_mean, y_min, y_max = generate_confidence_interval(y)
    plt.plot(x, y_mean, label=mylegend, linestyle=ls, linewidth=lw, c=color)
    plt.fill_between(x, y_min, y_max, alpha=transparency, color=color)
    return



seed = 4
step = 100
k=10
env = 'hopper'
plot_num = int(step/k)

returns = np.zeros([seed,step])

for i in range(seed):
    path = '/Users/wangxiyao/Downloads/exp_buffer/wmbpo/{}/{}_random_{}.txt'.format(env,env,i)
    f = open(path)
    j = 0
    line = f.readline()
    while line:
        if line.find('sum_reward') > 0:
            reward = float(line.split('sum_reward: ')[1].split(',')[0])
            returns[i][j] = float(reward)
            j += 1
        if j>= step:
            break
        else:
            line = f.readline()
    f.close()

mean = np.zeros([seed,plot_num])
for i in range(plot_num):
    for j in range(seed):
            mean[j][i] = np.mean(returns[j][i*k:(i+1)*k])


returns = np.zeros([seed,step])

for i in range(seed):
    path = '/Users/wangxiyao/Downloads/exp_buffer/wmbpo/{}/{}_reweight_model_dc96_{}.txt'.format(env,env,i)
    f = open(path)
    j = 0
    line = f.readline()
    while line:
        if line.find('sum_reward') > 0:
            reward = float(line.split('sum_reward: ')[1].split(',')[0])
            returns[i][j] = float(reward)
            j += 1
        if j>= step:
            break
        else:
            line = f.readline()
    f.close()

mean_ = np.zeros([seed,plot_num])
for i in range(plot_num):
    for j in range(seed):
            mean_[j][i] = np.mean(returns[j][i*k:(i+1)*k])



plot_ci(np.arange(0,plot_num), mean, seed, plot_num, 'MBPO', color='blue')
plot_ci(np.arange(0,plot_num), mean_, seed, plot_num, 'Ours', color='red')

plt.legend()
# plt.xticks([0,4.75,9.5,14.25,19],['0k','50k','100k','150k','200k'])
# plt.xticks([0,4.5,9],['0k','250k','500k'])
# plt.xticks([0,3,6,9],['0k','10k','20k','30k'])
# plt.xticks([0,4.5,9],['0k','2.5k','5k'])
plt.title('HalfCheetah')
plt.ylabel('Returns')
plt.xlabel('Steps')
plt.grid(True)
plt.show()
# tikzplotlib.save("halfcheetah.tex")