import os
import copy
import json
import sys
import numpy as np
import pickle
import orjson
import base64
from datetime import datetime
import torch


def extract_data_pair(data_dict_1, data_dict_2):
    if data_dict_2['seqname'] == 'default':
        return None
    else:
        e1 = data_dict_1['sense_data']['x_axis']
        e2 = data_dict_1['sense_data']['y_axis']
        e3 = data_dict_1['sense_data']['z_axis']
        global_pos_1 = data_dict_1['sense_data']['framepos']
        global_pos_2 = data_dict_2['sense_data']['framepos']
        global_disp = global_pos_2 - global_pos_1
        relative_disp = change_basis([e1, e2, e3], global_disp)[:2]
        data = {
            'acc': data_dict_1['sense_data']['acc'],
            'vel': data_dict_1['sense_data']['vel'],
            'gryo': data_dict_1['sense_data']['gyro'],
            'touch': data_dict_1['sense_data']['touch'],
            'joint_pos_1': data_dict_1['sense_data']['joint_pos'],
            'joint_vel': data_dict_1['sense_data']['joint_vel'],
            'target_1': data_dict_1['target'],
            'target_2': data_dict_2['target'],
            'disp_vect': relative_disp
        }
        return data


def load_txt(filepath):
    data = list()
    with open(filepath, 'r') as f:
        while f:
            line = f.readline()
            if not line:
                break
            data.append(b64_to_dict(line.strip('\n')))

    return data


def load_txt_limit(filepath, limit):
    with open(filepath, 'r') as f:
        line1 = f.readline()
        data1 = b64_to_dict(line1.strip('\n'))
        line2 = f.readline()
        data2 = b64_to_dict(line2.strip('\n'))
        init_data = tensorify_dict(extract_data_pair(data1, data2))
        list_dict = {key: list() for key in init_data.keys()}
        data_idx = 1
        while data_idx <= limit:
            line2 = f.readline()
            data1 = data2
            if not line2:
                break
            data2 = b64_to_dict(line2.strip('\n'))
            if data2['seqname'] != 'default':
                np_data = extract_data_pair(data1, data2)
                disp_mag = np.sqrt(np.sum(np_data['disp_vect'] ** 2))
                if disp_mag < 0.01:
                    if np.random.rand() < 0.95:
                        continue
                data = tensorify_dict(np_data)
                data_idx += 1
                if data_idx % 10000 == 0:
                    print(data_idx)
                for key in list_dict.keys():
                    list_dict[key].extend([data[key]])
        data_dict = dict()
        for key in list_dict.keys():
            data_dict[key] = torch.cat(list_dict[key], 0)
        return data_dict


def load_txt_fromto(filepath, fromline, toline):
    data = list()
    line_counter = 1
    with open(filepath, 'r') as f:
        while f:
            if line_counter < fromline:
                f.readline()
            else:
                line = f.readline()
                if not line or line_counter > toline:
                    break
                data.append(b64_to_dict(line.strip('\n')))
                line_counter += 1
    return data


def get_time_stamp():
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return timestamp


def change_basis(basis_vects, old_vect):
    return np.linalg.inv(np.array(basis_vects)).dot(old_vect)


def extract_dataset(data_list):
    init_data = tensorify_dict(extract_data_pair(data_list[0], data_list[1]))
    list_dict = {key: list() for key in init_data.keys()}
    for i in range(0, len(data_list)-1):
        data_dict = tensorify_dict(extract_data_pair(data_list[i], data_list[i+1]))
        for key in list_dict.keys():
            list_dict[key].extend([data_dict[key]])
    data_dict = dict()
    for key in list_dict.keys():
        data_dict[key] = torch.cat(list_dict[key], 0)
    return data_dict


def tensorify_dict(a_dict):
    t_dict = dict()
    for key in a_dict.keys():
        t_dict[key] = torch.tensor(a_dict[key]).unsqueeze(0).float()
    return t_dict


def project_u_onto_v(u, v):
    return np.sum(u*v)/np.sum(v ** 2)*v


def load_data(file_path, data_format, **kwargs):
    if type(file_path) is list:
        file_path = os.path.join(*file_path)

    if data_format == 'nparray':
        decoder = b64_to_np
    else:
        raise Exception('Invalid data_format type, must be one of ["nparray", "object"].')

    data = list()

    if 'start' in kwargs and 'end' in kwargs:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if kwargs['start'] <= i <= kwargs['end']:
                    data.append(decoder(line.strip('\n')))
    elif 'start' in kwargs:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if kwargs['start'] <= i:
                    data.append(decoder(line.strip('\n')))
    elif 'end' in kwargs:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i <= kwargs['end']:
                    data.append(decoder(line.strip('\n')))
    else:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(decoder(line.strip('\n')))

    return data


def parse_time(time_args):
    if 'sim_time' in time_args:
        time_parts = [i.strip() for i in time_args['sim_time'].split(',')]
        sim_time = 0
        for tstring in time_parts:
            time_unit = tstring[-1]
            time_amount = float(tstring[:-1])
            if time_unit == 's':
                sim_time += time_amount
            elif time_unit == 'm':
                sim_time += time_amount * 60
            elif time_unit == 'h':
                sim_time += time_amount * 60 * 60
            elif time_unit == 'd':
                sim_time += time_amount * 60 * 60 * 24
            else:
                raise Exception('Unit {} in {} not supported, must be one of ["s", "m", "h", "d"]'.format(time_unit, tstring))
        time_args['sim_time'] = int(sim_time)


    # T is the total time integrated over, from [0,T] inclusive
    # dt is the time-step of the integration
    # n is the number of points simulated, excluding the initial condition. This means n updates are evaluated
    # time_vector is the actual vector of times that we evaluate at
    if 'sim_time' in time_args and 'dt' in time_args:
        time_args['n'] = int(np.ceil(time_args['sim_time'] / time_args['dt']))
        time_args['sim_time'] = time_args['n'] * time_args['dt']
        # time_args['time_vector'] = np.linspace(0, time_args['sim_time'], time_args['n'] + 1)
        return time_args
    if 'sim_time' in time_args and 'n' in time_args:
        time_args['dt'] = time_args['sim_time'] / time_args['n']
        # time_args['time_vector'] = np.linspace(0, time_args['sim_time'], time_args['n']+1)
        return time_args
    if 'n' in time_args and 'dt' in time_args:
        time_args['sim_time'] = time_args['n'] * time_args['dt']
        # time_args['time_vector'] = np.linspace(0, time_args['sim_time'], time_args['n']+1)
        return time_args
    if 'time_vector' in time_args:
        raise Exception('Cannot handle time_vector')
        # return time_args


def obj_to_b64(obj):
    return base64.b64encode(orjson.dumps(obj.__dict__, option=orjson.OPT_SERIALIZE_NUMPY)).decode('ascii')


def dict_to_b64(dictionary):
    return base64.b64encode(orjson.dumps(dictionary, option=orjson.OPT_SERIALIZE_NUMPY)).decode('ascii')


def b64_to_dict(base_64):
    dictionary = orjson.loads(base64.b64decode(base_64))
    recursive_list_to_np(dictionary)
    return dictionary


def np_to_b64(nparray):
    return base64.b64encode(orjson.dumps(nparray, option=orjson.OPT_SERIALIZE_NUMPY)).decode('ascii')


def b64_to_np(base_64):
    return np.array(orjson.loads(base64.b64decode(base_64)))


def obj_to_hex(obj):
    return orjson.dumps(obj.__dict__, option=orjson.OPT_SERIALIZE_NUMPY).hex()


def dict_to_hex(dictionary):
    return orjson.dumps(dictionary, option=orjson.OPT_SERIALIZE_NUMPY).hex()


def hex_to_dict(hexidecimal):
    dictionary = orjson.loads(bytes.fromhex(hexidecimal))
    recursive_list_to_np(dictionary)
    return dictionary


def np_to_hex(nparray):
    return orjson.dumps(nparray, option=orjson.OPT_SERIALIZE_NUMPY).hex()


def hex_to_np(hexidecimal):
    return np.array(orjson.loads(bytes.fromhex(hexidecimal)))


def recursive_list_to_np(dictionary):
    for key, val in dictionary.items():
        if isinstance(val, list):
            dictionary[key] = np.array(val)
        elif isinstance(val, dict):
            recursive_list_to_np(val)


def serialize(data):
    return ' '.join([str(int(i)) for i in pickle.dumps(data)])


def deserialize(string):
    return pickle.loads(bytes([int(i) for i in string.split(' ')]))


def get_path(level):
    if level == 0:
        level = None
    else:
        level = -level
    return os.sep.join(os.path.abspath(__file__).split(os.sep)[0:level])


def dict_prod(key, vals, dict_list):
    dict_list_prod = []
    for val in vals:
        dict_list_copy = copy.deepcopy(dict_list)
        for dictionary in dict_list_copy:
            dictionary[key] = val
            dict_list_prod.append(dictionary)
    return dict_list_prod


def dict_factor(dictionary):
    dict_list = [copy.copy(dictionary)]
    for key in dictionary:
        vals = dictionary[key]
        dict_list = dict_prod(key, vals, dict_list)
    return dict_list


def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def try_parse_array_spec(arg):
    if type(arg) == str:
        args = arg.split(':')
        if len(args) == 3:
            if all([isfloat(a) for a in args]):
                start = float(args[0])
                increment = float(args[1])
                end = float(args[2])
                return list(np.around(np.arange(start, end + increment, increment), 4))
            else:
                return arg
        else:
            return arg
    else:
        return arg


def parse_config_dict(dictionary):
    for key, val in dictionary.items():
        if type(val) == dict:
            parse_config_dict(val)
        else:
            dictionary[key] = try_parse_array_spec(dictionary[key])

