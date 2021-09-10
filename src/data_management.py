import numpy as np
from io import StringIO
import os
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
import utils
import json
from utils import serialize
from utils import deserialize


def np_to_string(nparray):
    return ' '.join([str(i) for i in nparray])


def string_to_np(string):
    string.replace(' ', ',')


def stringify_state(state):
    return ','.join(stringify_varval(var, val) for var, val in state.items())


def stringify_varval(var, val):
    return '({}:[{}])'.format(var, ' '.join([str(i) for i in val]))


def average_nd_point(nd_array):
    return np.sum((1/nd_array.shape[0]) * nd_array, 0, keepdims=True)


def smooth_trajectory(trajectory, k):
    return np.concatenate([average_nd_point(trajectory[i:(i+k), :]) for i in range(trajectory.shape[0]-k)], axis=0)


def read_file(filepath, data_type):
    with open(filepath, 'r') as file:
        lines = [line.rstrip('\n') for line in file]
    if data_type == 'np as array':
        return np.transpose(np.concatenate([np.expand_dims(utils.b64_to_np(line), 1) for line in lines], axis=1))
        # return np.transpose(np.concatenate([np.expand_dims(np.loadtxt(StringIO(line)), 1) for line in lines], axis=1))
    elif data_type == 'np as list':
        return [utils.b64_to_np(line) for line in lines]
    elif data_type == 'dict as list':
        return [utils.b64_to_dict(line) for line in lines]
    elif data_type == 'obj as list':
        return [utils.b64_to_obj(line) for line in lines]
    else:
        raise Exception('Unsupported data-type output')


class DataScribe(object):
    def __init__(self, directory, config_name, store_in_dir):
        self.store_in_dir = store_in_dir
        self.config_name = config_name
        self.sim_id = '{}--[{}]'.format(self.config_name, datetime.now().strftime('%Y_%m_%d--%H_%M_%S'))
        if store_in_dir:
            self.directory = os.path.join(directory, self.sim_id)
            os.mkdir(self.directory)
        else:
            self.directory = directory

        self.data_structure = {
            'qpos': 'np as array',
            'qvel': 'np as array',
            # 'qacc': 'np as array',
            'ctrl': 'np as array',
            'snsr': 'np as array',
            # 'brain': 'obj as array',
            # 'maxuse_stack': 'np as array'
        }
        self.data_strings = {key: '' for key in self.data_structure.keys()}

        self.file_ids = self.data_structure.keys()
        if store_in_dir:
            self.file_paths = {key: os.path.join(self.directory, key) for key in self.file_ids}
        else:
            self.file_paths = {key: os.path.join(self.directory, '{}_{}'.format(key, self.sim_id)) for key in self.file_ids}
        self.files = {key: None for key in self.file_ids}
        self.pickle_file_path = os.path.join(self.directory, '{}.pickle'.format(self.sim_id))

    def open_files(self, mode):
        for file_id, file_path in self.file_paths.items():
            self.files[file_id] = open(file_path, mode)

    def close_files(self):
        for file_id, file_path in self.file_paths.items():
            self.files[file_id].close()
            self.files[file_id] = None

    def clean_up_temp_files(self):
        for file_id in self.file_ids:
            os.remove(self.file_paths[file_id])

    def close_and_package_files(self, metadata):
        if type(metadata) is dict:
            if self.store_in_dir:
                with open(os.path.join(self.directory, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=4)
                self.close_files()
            else:
                for file_id in self.file_ids:
                    metadata[file_id] = self.read_file(self.file_paths[file_id], self.data_structure[file_id])
                with open(self.pickle_file_path, 'wb') as handle:
                    pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.close_files()
                self.clean_up_temp_files()
        else:
            print('metadata must be a dictionary of values you want to save')

    def write_files(self, sim, brain):
        # state_string = ' '.join([str(i) for i in np.concatenate((sim.data.qpos[7:], sim.data.qvel[6:]))]) + '\n'
        self.data_strings['qpos'] = utils.np_to_b64(sim.data.qpos) + '\n'
        self.data_strings['qvel'] = utils.np_to_b64(sim.data.qvel) + '\n'
        # self.data_strings['qacc'] = utils.np_to_b64(sim.data.qacc) + '\n'
        self.data_strings['ctrl'] = utils.np_to_b64(sim.data.ctrl) + '\n'
        self.data_strings['snsr'] = utils.np_to_b64(sim.data.sensordata) + '\n'
        # self.data_strings['maxuse_stack'] = str(sim.data.maxuse_stack) + '\n'
        # self.data_strings['brain'] = utils.obj_to_b64(brain) + '\n'

        for file_id in self.file_ids:
            self.files[file_id].write(self.data_strings[file_id])


class DataProcessor(object):
    def __init__(self, file_path):
        self.data_path = file_path
        self.metadata = None
        self.data = None
        self.low_D_data = None
        self.smooth_data = None

    def read_data_folder(self):
        qpos = read_file(os.path.join(self.data_path, 'qpos'), 'np as array')[:, 7:]
        qvel = read_file(os.path.join(self.data_path, 'qvel'), 'np as array')[:, 6:]
        self.data = np.concatenate([qpos, qvel], axis=1)

    def read_text_file(self):
        self.data = read_file(self.data_path)
        with open(self.file_path, 'r') as file:
            lines = [line.rstrip('\n') for line in file]
        data = [np.expand_dims(np.ladtxt(StringIO(line)),1) for line in lines]
        self.data = np.transpose(np.concatenate(data, axis=1))

    def read_pickle_file(self):
        with open(self.data_path, 'rb') as handle:
            metadata = pickle.load(handle)
        self.metadata = metadata
        self.data = metadata['state_data']
        del metadata['state_data']

    def calculate_3D_projection(self):
        pca = PCA(n_components=3)
        # print(self.data.shape)
        self.smooth_data = smooth_trajectory(self.data, 10)
        pca.fit(self.smooth_data)
        self.low_D_data = pca.transform(self.smooth_data)

    def plot_data(self):
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(self.low_D_data[:, 0], self.low_D_data[:, 1], self.low_D_data[:, 2])
        ax.set_xlabel('PCA comp. 1')
        ax.set_ylabel('PCA comp. 2')
        ax.set_zlabel('PCA comp. 3')
        plt.title(self.data_path)
        plt.show()

    def view_data(self):
        self.read_data_folder()
        self.calculate_3D_projection()
        self.plot_data()

