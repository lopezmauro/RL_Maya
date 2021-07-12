import numpy as np
import os
import matplotlib.pyplot as plt


def plot_learning_courve(scores, figure_file, mean_amount=100):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-mean_amount):i+1])
    plt.plot(range(len(scores)), running_avg)
    plt.title('Running average of previous {} scores'.format(mean_amount))
    plt.savefig(figure_file)
    plt.show()


folder = r"D:\dev\RL_Maya\tests"
# all_subdirs = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
# latest_subdir = max(all_subdirs, key=os.path.getmtime)
latest_subdir = os.path.join(folder, '2021_07_08_18_22_goblin_mtx_collision_joint4_normalized')

"""
fileName = 'joint1_joint2_joint3_rwd.txt'
data_path = os.path.join(latest_subdir, fileName)
with open(data_path) as f:
    content = f.read().splitlines()
figure_file = '{}.png'.format(os.path.splitext(data_path)[0])
score = [float(a) for a in content]
plot_learning_courve(score, figure_file, mean_amount=1)

"""
fileName = '_agnt_rew.txt'
found_files = [a for a in os.listdir(latest_subdir) if a.endswith(fileName)]
if not found_files:
    raise BaseException(f"unable to find {fileName} on {latest_subdir}")
data_path = os.path.join(latest_subdir, found_files[0])
with open(data_path) as f:
    content = f.read().splitlines()
locData = dict()
for data in content:
    parts = data.split(" : ")
    name = parts[-1]
    value = float(parts[0])
    locData.setdefault(name, []).append(value)
i = 0
batch = 5
for name, values in locData.items():
    print(name, values)
    plt.plot(range(len(values)), values, label=name)
    plt.legend()
    plt.show()
    figure_file = '{}.png'.format(name)
    plt.savefig(os.path.join(latest_subdir, figure_file))
    plt.clf()
