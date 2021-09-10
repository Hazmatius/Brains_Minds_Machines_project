import torch
import numpy as np
import utils
import os
import matplotlib.pyplot as plt


def plot_axes(x_framepos, y_framepos, x_axis_list, y_axis_list, idx):
    x1 = x_framepos[idx - 1]
    y1 = y_framepos[idx - 1]
    x_x2 = x1 + x_axis_list[idx - 1][0]
    x_y2 = y1 + x_axis_list[idx - 1][1]

    y_x2 = x1 + y_axis_list[idx - 1][0]
    y_y2 = y1 + y_axis_list[idx - 1][1]

    plt.plot([x1, x_x2], [y1, x_y2], color=[1, 0, 0])
    plt.plot([x1, y_x2], [y1, y_y2], color=[0, 1, 0])


def plot_dir(x_pos_list, y_pos_list, x_dir_list, y_dir_list, idx):
    x1 = x_pos_list[idx]
    y1 = y_pos_list[idx]

    x2 = x1 + x_dir_list[idx]
    y2 = y1 + y_dir_list[idx]

    plt.plot([x1, x2], [y1, y2], color=[1, 0, 0])


# extract data:
data_file = os.path.join('/Volumes/ALEX_SSD/HNSC_data', '{}.txt'.format('target_1629085217.862489'))
data_list = list()
counter = 0
with open(data_file, 'r') as f:
    line = True
    while line:
        line = f.readline()
        if not line:
            break
        counter += 1
        data_dict = utils.b64_to_dict(line.strip('\n'))
        data_list.append(data_dict)

x_framepos = list()
y_framepos = list()
x_c_framepos = list()
y_c_framepos = list()
start_x = 0
start_y = 0
x_axis_list = list()
y_axis_list = list()

x_dir_list = list()
y_dir_list = list()
x_pos_list = list()
y_pos_list = list()

for i in range(len(data_list)-1):
    data_dict_1 = data_list[i]
    data_dict_2 = data_list[i+1]

    x_dir_list.append(data_dict_1['sense_data']['dir'][0])
    y_dir_list.append(data_dict_1['sense_data']['dir'][1])
    x_pos_list.append(data_dict_1['sense_data']['pos'][0])
    y_pos_list.append(data_dict_1['sense_data']['pos'][1])

    e1 = data_dict_1['sense_data']['x_axis']
    e2 = data_dict_1['sense_data']['y_axis']
    e3 = data_dict_1['sense_data']['z_axis']
    W = np.array([e1, e2, e3])
    x_axis_list.append(e1)
    y_axis_list.append(e2)
    framepos = data_dict_1['sense_data']['framepos']
    x_framepos.append(framepos[0])
    y_framepos.append(framepos[1])
    disp_vect = data_dict_2['sense_data']['framepos'] - data_dict_1['sense_data']['framepos']
    x_c_framepos.append(start_x)
    y_c_framepos.append(start_y)
    rel_disp_vect = np.linalg.inv(W).dot(disp_vect)
    start_x += rel_disp_vect[0]
    start_y += rel_disp_vect[1]
    # print(rel_disp_vect)

start_idx = 0
end_idx = 100

plt.plot(x_pos_list[start_idx:end_idx], y_pos_list[start_idx:end_idx], '-o', color=[0, 0, 0])

for i in range(start_idx, end_idx, 1):
    plot_dir(x_pos_list, y_pos_list, x_dir_list, y_dir_list, i)
# plt.plot(x_framepos[start_idx:end_idx], y_framepos[start_idx:end_idx], '-o', color=[0, 0, 0])

# for i in range(start_idx, end_idx, 6):
    # plot_axes(x_framepos, y_framepos, x_axis_list, y_axis_list, i)


plt.gca().set_aspect('equal')
plt.show()