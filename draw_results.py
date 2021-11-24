import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import os


def draw_results():
    # name_list = ['exp_hop_wei98', 'exp_hop_wei97', 'exp_hop_0_reweight', 'exp_hop_0_ori']
    # name_list = ['exp_wal_0_ori', 'exp_wal_0_reweight', ]
    # name_list = ['exp_hop_rb10k', 'exp_hop_rb100k', 'exp_hop_0_ori']
    # name_list = ['exp_hop_dc_dc', 'exp_hop_no_dc',  'exp_hop_0_ori'] #'exp_hop_dc_no', 'exp_hop_no_no',
    name_list = ['exp_hop_ori', 'exp_hop_dc_dc_97', ] #'exp_hop_dc_no', 'exp_hop_no_no',
    color_list = ['b', 'r', 'g', 'cyan', 'magenta', 'yellow', ]
    savename = 'tmp'
    ax = plt.gca()

    if not os.path.isdir('./results/'):
        os.makedirs('./results/')

    for i,name in enumerate(name_list):
        f = open('../exp/' + name + '/log.txt') # './exp' will become larger as experiments increasing, remove it from code.

        step_list = []
        reward_list = []

        line = f.readline()
        while line:
            if line.find('sum_reward') > 0:
                step = float(line.split('sum_reward:')[0].split('total_step:')[1].split(',')[0])
                reward = float(line.split('sum_reward:')[1].split('\n')[0])
                step_list.append(step)
                reward_list.append(reward)
            line = f.readline()
        f.close()

        ax.scatter(np.array(step_list), np.array(reward_list), color=color_list[i], s=1, label=name)
        # ax.plot(np.array(step_list), np.array(reward_list), color=color_list[i], label=name)

    plt.legend()
    plt.savefig('./results/'+savename+'.png')
    plt.show()


if __name__ == '__main__':
    draw_results()
