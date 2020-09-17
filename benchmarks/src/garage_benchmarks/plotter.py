import os

import matplotlib.pyplot as plt
import numpy as np

from helper import _read_csv, _read_txt

seeds = [27, 64, 74]  # potentially change this to match your seed values.
xcolumn = 'TotalEnvSteps'
xlabel = 'Total Environment Steps'
ycolumn = 'Evaluation/AverageReturn'
ylabel = 'Average Return'

# Mujoco1M envs by default, change if needed.
# env_ids = [
#     'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'HalfCheetah-v2',
#     'Hopper-v2', 'Walker2d-v2', 'Reacher-v2', 'Swimmer-v2'
# ]

env_ids = [
    'HalfCheetah-v2', 'Hopper-v2', 'InvertedDoublePendulum-v2', 'Walker2d-v2'
]

plot_dir = '/home/resl/iris/plots/Sept04'  # make sure this dir exists, the script doesn't create it
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

for env_id in env_ids:

    # change these directories to point to your benchmarks
    # dir = [
    #     '/home/resl/iris/garage/data/local/benchmarks/td3_benchmarks/td3-garage-tf_' +
    #     env_id + '_',
    #     '/home/resl/iris/garage/data/local/benchmarks/td3_benchmarks_1/td3-garage-pytorch_'
    #     + env_id + '_', '/home/resl/iris/garage/data/local/benchmarks/td3_spinningup/' + env_id + '_pytorch/' + env_id + '_pytorch'
    # ]
    dir = [
        # '/home/resl/iris/garage/data/local/benchmarks/td3_benchmarks/td3-garage-tf_' +
        # env_id + '_',
        '/home/resl/iris/garage/data/local/benchmarks/td3_benchmarks/td3-garage-pytorch_'
        + env_id + '_', '/home/resl/iris/garage/data/local/benchmarks/td3_spinningup/' + env_id + '_pytorch/' + env_id + '_pytorch'
    ]
    # labels = ['td3-garage-tf', 'td3-garage-torch', 'td3-openai-baselines']
    labels = ['td3-garage-torch', 'td3-openai-spinningup']

    _plot = {}

    if _plot is not None and env_id not in _plot:
        _plot[env_id] = {'xlabel': xlabel, 'ylabel': ylabel}

    plt.figure(env_id)
    for label, subdir in zip(labels[:-1], dir[:-1]):

        task_ys = []
        for seed in seeds:
            xs, ys = _read_csv(subdir + str(seed), xcolumn, ycolumn)
            task_ys.append(ys)

        ys_mean = np.array(task_ys).mean(axis=0)
        ys_std = np.array(task_ys).std(axis=0)

        plt.plot(xs, ys_mean, label=label)
        plt.fill_between(xs, (ys_mean - ys_std), (ys_mean + ys_std), alpha=.1)

    baselines_seeds = [27, 64, 74]

    task_ys = []
    for seed in baselines_seeds:
        # xs, ys = _read_csv(dir[-1] + '_s' + str(seed), 'total_timesteps',
        #                    'eprewmean')
        xs, ys = _read_txt(dir[-1] + '_s' + str(seed), 'TotalEnvInteracts')
        task_ys.append(ys)

    ys_mean = np.array(task_ys).mean(axis=0)
    ys_std = np.array(task_ys).std(axis=0)

    plt.plot(xs, ys_mean, label=labels[-1])
    plt.fill_between(xs, (ys_mean - ys_std), (ys_mean + ys_std), alpha=.1)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(env_id)
    plt.xlim([0, 1.0e6])
    plt.savefig(plot_dir + '/' + env_id)
    print(plot_dir + '/' + env_id)