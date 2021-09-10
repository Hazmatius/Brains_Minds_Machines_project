import numpy as np
import copy
import random
import torch
import torch.nn as nn
import utils
import pickle
import matplotlib.pyplot as plt


def extract_inputs(minibatch):
    inputs = ['acc', 'vel', 'gryo', 'touch', 'joint_pos_1', 'joint_vel', 'target_1', 'target_2']
    return torch.cat([minibatch[key] for key in inputs], dim=1)


def get_minibatch(dataset, batch_size):
    minibatch = dict()
    idxs = np.random.choice([i for i in range(dataset['acc'].shape[0])], size=batch_size, replace=False)
    for key in dataset.keys():
        minibatch[key] = dataset[key][idxs, :]
    return minibatch


def permute_target(vect, perm):
    return copy.copy(vect)[perm]


def rand_permute_sequence(seq):
    # p1 = [0, 1, 2, 3, 4, 5, 6, 7]
    p2 = [2, 3, 4, 5, 6, 7, 0, 1]
    p3 = [6, 7, 0, 1, 2, 3, 4, 5]
    p4 = [4, 5, 6, 7, 0, 1, 2, 3]
    rand_perm = random.choice([p2, p3, p4])
    new_seq = [permute_target(target, rand_perm) for target in seq]
    return new_seq


def calc_distances(disp_vector, dirs):
    signed_dist = dict()
    for direction in dirs.keys():
        signed_dist[direction] = np.dot(disp_vector, dirs[direction]) / np.linalg.norm(dirs[direction])
    return signed_dist


def get_dir(sensor_data, direction):
    x = sensor_data['x_axis']
    y = sensor_data['y_axis']

    if direction == 0:
        return get_unit(utils.change_basis([x, y], x))
    elif direction == 1:
        return get_unit(utils.change_basis([x, y], x + y))
    elif direction == 2:
        return get_unit(utils.change_basis([x, y], y))
    elif direction == 3:
        return get_unit(utils.change_basis([x, y], -x + y))
    elif direction == 4:
        return get_unit(utils.change_basis([x, y], -x))
    elif direction == 5:
        return get_unit(utils.change_basis([x, y], -x - y))
    elif direction == 6:
        return get_unit(utils.change_basis([x, y], -y))
    elif direction == 7:
        return get_unit(utils.change_basis([x, y], x - y))


def get_dirs(sensor_data):
    x = sensor_data['x_axis']
    y = sensor_data['y_axis']
    dirs = {
        0: get_unit(x),
        1: get_unit(x + y),
        2: get_unit(y),
        3: get_unit(-x + y),
        4: get_unit(-x),
        5: get_unit(-x - y),
        6: get_unit(-y),
        7: get_unit(x - y)
    }
    return dirs


def parse_sensor_data(sim_obj):
    sensor_data = {
        'x_axis': sim_obj.data.sensordata[0:3],
        'y_axis': sim_obj.data.sensordata[3:6],
        'z_axis': sim_obj.data.sensordata[6:9],
        'acc': sim_obj.data.sensordata[9:12],
        'gyro': sim_obj.data.sensordata[12:15],
        'vel': sim_obj.data.sensordata[15:18],
        'joint_pos': sim_obj.data.sensordata[18:26],
        'joint_vel': sim_obj.data.sensordata[26:34],
        'touch': sim_obj.data.sensordata[34:38],
        'framepos': sim_obj.data.sensordata[38:41]
    }
    sensor_data['dir'] = get_unit(sensor_data['x_axis'])
    sensor_data['pos'] = sensor_data['framepos'][0:2]
    return sensor_data


def project_u_onto_v(u, v):
    return np.sum(u*v)/np.sum(v ** 2)*v


def get_unit(array):
    xy_array = array[:2]
    xy_array = xy_array / np.sqrt(np.sum(xy_array ** 2))
    return xy_array


def default():
    return np.array([0, 70, 0, -70, 0, -70, 0, 70]) * np.pi / 180


def mutate_sequence(sequence):
    # 0: mutate node
    # 1: add node
    # 2: remove node

    new_sequence = copy.copy(sequence['sequence'])
    # if np.random.rand() < .9:
    #     new_sequence = rand_permute_sequence(new_sequence)

    if len(sequence['sequence']) > 2:
        option = random.choices([0, 1, 2], weights=[.6, .2, .2], k=1)[0]
    else:
        option = random.choices([0, 1], weights=[.7, .3], k=1)[0]

    if option == 0:
        idx = random.choice([i for i in range(len(sequence['sequence']))])
        new_sequence[idx] = mutate_target(new_sequence[idx])
    elif option == 1:
        idx = random.choice([i for i in range(len(sequence['sequence'])+1)])
        new_sequence.insert(idx, None)
        if idx == 0:
            avg_target = (new_sequence[1] + new_sequence[-1])/2
        elif idx == len(new_sequence)-1:
            avg_target = (new_sequence[0] + new_sequence[-2])/2
        else:
            avg_target = (new_sequence[idx-1] + new_sequence[idx+1])/2
        new_sequence[idx] = mutate_target(avg_target)
    elif option == 2:
        idx = random.choice([i for i in range(len(sequence['sequence']))])
        del new_sequence[idx]

    return new_sequence


def mutate_target(target):
    min_theta = np.array([-30, 30, -30, -70, -30, -70, -30, 30]) * np.pi / 180
    max_theta = np.array([30, 70, 30, -30, 30, -30, 30, 70]) * np.pi / 180
    if np.random.rand() < 0.5:
        idx = random.choice([i for i in range(8)])
        new_target = target
        new_target[idx] + np.random.randn(1) / 10
    else:
        mutation = np.random.randn(8) / 10
        new_target = target + mutation
    outof = new_target < min_theta
    new_target[outof] = min_theta[outof]
    outof = new_target > max_theta
    new_target[outof] = max_theta[outof]
    return new_target


def rand_target():
    min_theta = np.array([-30, 30, -30, -70, -30, -70, -30, 30])
    max_theta = np.array([30, 70, 30, -30, 30, -30, 30, 70])
    return np.random.uniform(min_theta * np.pi / 180, max_theta * np.pi / 180)


class Logger(object):
    """
    Class for handling logging, allows to silence messages without deleting them
    """
    def __init__(self):
        self.level = -np.inf  # any message with a level above this gets printed
        # making the level higher will exclude more messages
        # making a messages level lower will decrease it's priority for the logger

    def set_level(self, level):
        self.level = level

    def log(self, msg, level):
        if level > self.level:
            print(msg)


class SYM(object):
    """
    Contains the full stack of controllers, including:
    pth_controller, which manages the navigation and exploration of 2D space
    way_controller, which manages the movement from point to point
    seq_controller, which manages the learning and execution of limb-position sequences
    tar_controller, which manges the transitions between limb-positions
    pid_controller, which is a PID controller
    """
    def __init__(self, dt, sim):
        self.dt = dt
        self.time = 0

        self.ctrl = np.zeros(8)
        self.sense_data = parse_sensor_data(sim)

        self.log = Logger()
        self.log.set_level(3)

        self.pth_controller = PTH(self, self.log)
        self.way_controller = WAY(self, self.log)
        self.seq_controller = SEQ(self, self.log)
        self.tar_controller = TAR(self, self.log)
        self.pid_controller = PID(8, 70, 30, 1, dt, 0.1, dt, 10, self.log)

    def initialize(self):
        self.pth_controller.initialize()
        self.way_controller.initialize()
        self.seq_controller.initialize()

    def sense(self, x):
        self.sense_data = x

    def process(self):
        self.time += self.dt
        x = self.sense_data['joint_pos']

        self.pth_controller.process()
        self.way_controller.process()
        self.seq_controller.process()
        self.tar_controller.process()
        self.ctrl = self.pid_controller.process(x)

        return self.ctrl

    def get_save_data(self):
        data = {
            'time': self.time,
            'sense_data': self.sense_data,
            'target': self.pid_controller.target,
            'control': self.ctrl,
            'idx': self.tar_controller.target_idx,
            'seqname': self.tar_controller.sequence['name']
        }
        return data


class PTH(object):
    """
    PTH ('path') controller manages the navigation and exploration of space. It does this by maintaining a graph
    representation of the points it has visited.
    """
    def __init__(self, controller, logger):
        self.log = logger
        self.controller = controller
        self.spatial_memory = Graph()
        self.mode = 'explore'
        self.waypoint = None
        self.last_waypoint = None
        self.next_waypoint = None

    def initialize(self):
        self.waypoint = Node(self.controller.sense_data['pos'], -1)
        self.spatial_memory.add_node(self.waypoint)
        self.waypoint.set_reached(1)
        self.next_waypoint = self.waypoint
        self.last_waypoint = self.waypoint
        self.controller.way_controller.set_waypoint_seq([self.waypoint])

    def go_to_near(self):
        next_pos = self.spatial_memory.get_rand_pos_from_node(self.waypoint)
        if next_pos is None:
            return 1
        self.next_waypoint = Node(next_pos, -1)
        self.spatial_memory.add_node(self.next_waypoint)
        wp_seq = [self.waypoint, self.next_waypoint]
        self.controller.way_controller.set_waypoint_seq(wp_seq)
        return 0

    def go_to_random(self):
        next_pos, end_node = self.spatial_memory.get_near_rand_pos(self.controller.sense_data['pos'])
        if next_pos is None:
            end_node = self.spatial_memory.get_rand_node()
            wp_seq = self.spatial_memory.traverse(self.waypoint, end_node)
            self.controller.way_controller.set_waypoint_seq(wp_seq)
        else:
            wp_seq = self.spatial_memory.traverse(self.waypoint, end_node)
            self.next_waypoint = Node(next_pos, -1)
            self.spatial_memory.add_node(self.next_waypoint)
            wp_seq.append(self.next_waypoint)
            self.controller.way_controller.set_waypoint_seq(wp_seq)

    def process(self):
        if self.controller.way_controller.success == 1:  # successfully reached end of waypoint sequence

            if self.mode == 'explore':
                if np.random.rand() < 1:  # pick a waypoint that is near to where we are
                    if self.go_to_near() == 1:
                        self.go_to_random()
                else:
                    self.go_to_random()

        elif self.controller.way_controller.success == -1:  # we failed to reach the end of the waypoint sequence
            # print('WAY:FAIL')
            # self.waypoint = self.spatial_memory.get_nearest_node(self.controller.sense_data['pos'])
            print('FALLBACK TO {}'.format(self.last_waypoint.idx))
            self.waypoint = self.last_waypoint
            next_pos, self.next_waypoint = self.spatial_memory.get_near_rand_pos(self.controller.sense_data['pos'])
            # self.next_waypoint = self.spatial_memory.get_rand_node()
            wp_seq = self.spatial_memory.traverse(self.waypoint, self.next_waypoint)
            self.controller.way_controller.set_waypoint_seq(wp_seq)


class WAY(object):
    """
    WAY ('waypoint') controller manages the movement from one waypoint to another by exeucting waypoint sequences.
    When it has reached the target waypoint, it emits a message indicating that it has succeeded. WAY will link points
    together if one was reachable from the other, and will mark two points as "disconnected" when the controller fails
    to reach one waypoint from another.
    """
    def __init__(self, controller, logger):
        self.log = logger
        self.controller = controller
        self.waypoint_sequence = None
        self.success = 0
        self.wpseq_idx = 0
        self.plot_counter = 0

    def initialize(self):
        self.manage_way()

    def set_waypoint_seq(self, waypoint_seq):
        self.waypoint_sequence = waypoint_seq
        self.wpseq_idx = 0
        self.success = 0
        self.manage_way()

    def manage_way(self):
        self.controller.seq_controller.set_waypoint(self.waypoint_sequence[self.wpseq_idx])

    def process(self):
        if self.controller.seq_controller.success == 1:  # we have gone from one waypoint to another
            if self.plot_counter % 50 == 0:
                self.controller.pth_controller.spatial_memory.plot_graph()
            self.plot_counter += 1
            if self.wpseq_idx > 0:
                start_waypoint = self.waypoint_sequence[self.wpseq_idx - 1]
                end_waypoint = self.waypoint_sequence[self.wpseq_idx]
                self.controller.pth_controller.waypoint = end_waypoint
                end_waypoint.set_reached(1)
                start_waypoint.connect_to(end_waypoint, 1)
                self.controller.pth_controller.spatial_memory.add_node(end_waypoint)
                self.controller.pth_controller.waypoint = end_waypoint
            self.wpseq_idx += 1
            if self.wpseq_idx < len(self.waypoint_sequence):
                self.manage_way()
            else:
                self.success = 1
        elif self.controller.seq_controller.success == -1:  # we have failed to get to the next waypoint
            print('WAY:FAILED')
            self.success = -1
            start_waypoint = self.waypoint_sequence[self.wpseq_idx - 1]
            start_waypoint.fails += 1
            end_waypoint = self.waypoint_sequence[self.wpseq_idx]
            if self.wpseq_idx > 0:
                end_waypoint.set_reached(-1)
                start_waypoint.connect_to(end_waypoint, -1)
            self.controller.pth_controller.spatial_memory.add_node(end_waypoint)
            self.controller.pth_controller.last_waypoint = self.waypoint_sequence[self.wpseq_idx - 1]


class SEQ(object):
    """
    The SEQ ('sequence') controller has two important jobs. First, given a direction to travel in, it selects the
    sequence that will move it in that direction fastest. Second, through the use of a genetic algorithm that can be
    optionally enhanced with a neural network, it can also evolve new sequences and select the ones that perform best.
    """
    def __init__(self, controller, logger):
        self.controller = controller
        self.log = logger
        self.seq_mem_dict = {i: list() for i in range(8)}  # dictionary of lists of sequences, organized by direction
        self.seq_queue = list()  # queue of sequences to be executed
        self.starting_dirs = None
        self.starting_point = None
        self.starting_time = None
        self.counter = 0
        self.speeds = np.zeros(8)
        self.max_seq_itr = 5
        self.mode = 'navigate'
        self.waypoint = None
        self.success = 0
        self.timer = 0
        self.expected_time = 0

    def set_waypoint(self, waypoint):
        self.waypoint = waypoint
        distance = np.sqrt(np.sum((self.controller.sense_data['pos'] - self.waypoint.pos) ** 2))
        self.expected_time = 10 * (distance + 1.5) / np.min(self.speeds)
        self.success = 0
        self.timer = 0

    def load_seq_mem(self, filepath):
        with open(filepath, 'rb') as f:
            self.seq_mem_dict = pickle.load(f)
        self.find_fastest_speeds()

    def choose_random_sequence(self):
        # choose least-evolved direction:
        direction = np.argmin(self.speeds)
        self.target_direction = direction
        if len(self.seq_mem_dict[direction]) > 0:
            seq_speeds = [seq['speeds'][direction] for seq in self.seq_mem_dict[direction]]
            return random.choices(self.seq_mem_dict[direction], weights=seq_speeds, k=1)[0]
        else:
            return self.generate_random_sequence()

    def generate_mutant_sequence(self, sequence):
        new_sequence = mutate_sequence(sequence)
        seq_dict = {
            'name': str(self.counter),
            'sequence': new_sequence,
            'dist': None
        }
        self.counter += 1
        return seq_dict

    def generate_random_sequence(self):
        sequence = list()
        for i in range(np.random.randint(2, 7)):
            sequence.append(rand_target())
        seq_dict = {
            'name': str(self.counter),
            'sequence': sequence,
            'dist': None
        }
        self.counter += 1
        return seq_dict

    def generate_default_sequence(self):
        seq_dict = {
            'name': 'default',
            'sequence': [default()],
            'dist': None
        }
        return seq_dict

    def get_fastest_sequence(self, direction):
        if len(self.seq_mem_dict[direction]) == 0:
            new_sequence = self.generate_random_sequence()
            self.counter -= 1
            return new_sequence
        return self.seq_mem_dict[direction][-1]

    def find_fastest_speeds(self):
        for i in range(8):
            self.speeds[i] = self.get_fastest_sequence(i)['speeds'][i]

    def evolve(self):
        if len(self.seq_queue) < 10:
            if np.random.rand() < 0.8:  # mutate
                selected_sequence = self.choose_random_sequence()
                new_sequence = self.generate_mutant_sequence(selected_sequence)
                self.seq_queue.append(new_sequence)
                self.seq_queue.append(self.generate_default_sequence())
            else:  # random
                new_sequence = self.generate_random_sequence()
                self.seq_queue.append(new_sequence)
                self.seq_queue.append(self.generate_default_sequence())

    def forget(self):
        for i in range(8):
            if len(self.seq_mem_dict[i]) > 20:
                self.log.log(1*'....' + 'SEQ:forget', 0)
                self.seq_mem_dict[i] = self.seq_mem_dict[i][1:]

    def calculate_speed(self, sequence):
        """
        Calculates the speed in each direction for a given sequence
        If the sequence was faster in any direction than a previous record, the sequence is added to memory
        :param sequence:
        :return:
        """
        ending_point = copy.copy(self.controller.sense_data['pos'])
        ending_time = self.controller.time
        displacement = ending_point - sequence['starting_point']
        signed_distances = calc_distances(displacement, sequence['starting_dirs'])
        time_diff = ending_time - sequence['starting_time']
        signed_speeds = {k: signed_distances[k] / time_diff for k in signed_distances.keys()}
        sequence['dists'] = signed_distances
        sequence['speeds'] = signed_speeds
        speed_updates = [False for i in range(8)]
        for k in signed_speeds.keys():
            if signed_speeds[k] > self.speeds[k]:
                speed_updates[k] = True
                self.speeds[k] = signed_speeds[k]
                self.seq_mem_dict[k].append(sequence)
            self.speeds[k] = max(self.speeds[k], signed_speeds[k])
        if any(speed_updates):
            print(20*'=')
            for k in range(8):
                if speed_updates[k]:
                    print('| {} : {:.3f} *'.format(k, round(self.speeds[k], 4)))
                else:
                    print('| {} : {:.3f}'.format(k, round(self.speeds[k], 4)))
            print(20*'=')

    def manage_seq(self, sequence):
        self.controller.tar_controller.set_sequence(sequence)
        self.controller.tar_controller.success = 0

    def get_waypoint_direction(self):
        """
        Calculates the direction [0, 7] that the waypoint is in
        :return: direction
        """
        way_point_disp_vect = self.waypoint.pos - self.controller.sense_data['pos']
        if np.sqrt(np.sum(way_point_disp_vect ** 2)) < 1.5:
            self.success = 1
        else:
            self.success = 0
        signed_distances = calc_distances(way_point_disp_vect, get_dirs(self.controller.sense_data))
        vect_signed_distances = np.zeros(8)
        for k in range(8):
            vect_signed_distances[k] = signed_distances[k]
        direction = np.argmax(vect_signed_distances)
        return direction

    def pop_seq_from_queue(self):
        next_sequence = self.seq_queue.pop(0)
        next_sequence['starting_dirs'] = get_dirs(self.controller.sense_data)
        next_sequence['starting_point'] = copy.copy(self.controller.sense_data['pos'])
        next_sequence['starting_time'] = self.controller.time
        return next_sequence

    def initialize(self):
        self.starting_dirs = get_dirs(self.controller.sense_data)
        self.starting_point = copy.copy(self.controller.sense_data['pos'])
        self.starting_time = self.controller.time
        self.evolve()
        next_sequence = self.pop_seq_from_queue()
        self.manage_seq(next_sequence)

    def process(self):
        self.timer += self.controller.dt
        if self.mode == 'explore':
            if self.controller.tar_controller.success == 1:  # we have completed a sequence some number of times
                self.log.log(1*'....' + 'SEQ:success on seq.{}'.format(self.controller.tar_controller.sequence['name']), 0)
                if self.controller.tar_controller.sequence['name'] != 'default':
                    self.calculate_speed(self.controller.tar_controller.sequence)
                next_sequence = self.pop_seq_from_queue()
                self.forget()
                self.evolve()

                self.manage_seq(next_sequence)
        if self.mode == 'navigate':
            if self.timer > self.expected_time:
                # print('{} : {}'.format(self.timer, self.expected_time))
                self.success = -1
            if self.controller.tar_controller.success == 1:
                direction = self.get_waypoint_direction()
                sequence = self.get_fastest_sequence(direction)
                self.manage_seq(sequence)


class TAR(object):
    """
    Executes a sequence of targets in order. Once a target has been reached (as determined by the PID controller), the
    TAR controller will set the PID setpoint to the next target in it's sequence.
    """
    def __init__(self, controller, logger):
        self.controller = controller
        self.log = logger
        self.sequence = None
        self.target_idx = 0
        self.seq_itr = 0
        self.max_seq_itr = 5
        self.success = 0
        self.intuition = None

    def set_sequence(self, sequence):
        self.succesds = 0
        self.sequence = sequence
        self.target_idx = 0
        self.seq_itr = 0
        self.manage_tar()

    def manage_tar(self):
        self.controller.pid_controller.set_target(self.sequence['sequence'][self.target_idx])
        self.controller.pid_controller.reset_timer()
        # self.controller.pid_controller.success = 0

    def intuit_next_target(self, next_target, target_direction):
        sense_data = self.controller.sense_data
        target_tensor = torch.tensor(get_dir(sense_data, target_direction)).unsqueeze(0).float()
        np_dict = {
            'acc': sense_data['acc'],
            'vel': sense_data['vel'],
            'gryo': sense_data['gyro'],
            'tourch': sense_data['touch'],
            'joint_pos_1': sense_data['joint_pos'],
            'joint_vel': sense_data['joint_vel'],
            'target_1': sense_data['joint_pos'],
            'target_2': torch.tensor(next_target).unsqueeze(0).float()
        }
        tensor_dict = utils.tensorify_dict(np_dict)
        optim = torch.optim.SGD([tensor_dict['target_2']], lr=0.1)
        criterion = nn.MSELoss()
        for i in range(1):
            output_direction = self.intuition.process(tensor_dict)
            loss = criterion(output_direction, target_tensor)
            optim.zero_grad()
            loss.backward()
            optim.step()

        new_target = tensor_dict['target_2'].detach().numpy()
        min_theta = np.array([-30, 30, -30, -70, -30, -70, -30, 30]) * np.pi / 180
        max_theta = np.array([30, 70, 30, -30, 30, -30, 30, 70]) * np.pi / 180
        outof = new_target < min_theta
        new_target[outof] = min_theta[outof]
        outof = new_target > max_theta
        new_target[outof] = max_theta[outof]
        return new_target

    def process(self):
        if self.sequence is not None:
            if self.sequence['name'] == 'default':
                if self.seq_itr > 1:
                    self.success = 1
            if self.seq_itr >= self.max_seq_itr:
                self.success = 1
            if self.controller.pid_controller.success == 1:
                self.target_idx += 1
                if self.target_idx >= len(self.sequence['sequence']):
                    self.log.log(2 * '....' + 'TAR:success on seq.{}'.format(self.sequence['name']), 0)
                    self.target_idx = 0
                    self.seq_itr += 1
                # print(self.seq_itr)
                self.manage_tar()
            elif self.success == -1:
                new_target = default()
                self.controller.pid_controller.set_target(new_target)
                self.controller.pid_controller.reset_timer()
                self.controller.pid_controller.reset_integral()


class PID(object):
    """
    PID controller generates forces to move a controlled variable towards a setpoint
    """
    def __init__(self, dim, P, I, D, dt, s_T, t_T, e_T, logger):
        self.log = logger
        self.P = P
        self.I = I
        self.D = D
        self.dt = dt
        self.dim = dim
        self.error_t1 = np.zeros(dim)
        self.error_t2 = np.zeros(dim)
        self.integral = np.zeros(dim)
        self.target = np.zeros(dim)
        self.s_T = s_T  # space threshold
        self.t_T = t_T  # time threshold
        self.e_T = e_T  # integral error threshold
        self.timer = 0
        self.success = 0

    def set_space_threshold(self, threshold):
        self.s_T = threshold

    def set_time_threshold(self, threshold):
        self.t_T_ = threshold

    def set_error_threshold(self, threshold):
        self.e_T = threshold

    def set_target(self, target):
        """
        Sets the setpoint of the PID controller
        :param target:
        :return:
        """
        self.target = target
        self.success = 0

    def reset_timer(self):
        self.timer = 0

    def reset_integral(self):
        self.integral = np.zeros(self.dim)

    def check_success(self):
        """
        Checks of the PID controlled variable has reached the setpoint
        :return: 1 for success, -1 for failure, 0 for no success
        """
        if np.sqrt(np.sum(self.error_t2 ** 2)) < self.s_T:
            self.timer += 1
        if np.all(np.abs(self.integral) < self.e_T) and self.timer * self.dt > self.t_T:
            self.log.log(3*'....' + 'PID:success', 0)
            return 1
        elif np.any(np.abs(self.integral) > self.e_T):
            return -1
        else:
            return 0

    def process(self, x):
        """
        Calculates the error between the controlled variable and the setpoint, generating forces to minimize the error.
        :param x:
        :return:
        """
        self.error_t1 = self.error_t2
        self.error_t2 = self.target - x
        d_error_dt = (self.error_t2 - self.error_t1) / self.dt
        self.integral += self.error_t2 * self.dt
        p_term = self.P * self.error_t2
        i_term = self.I * self.integral
        d_term = self.D * d_error_dt
        self.success = self.check_success()
        u = p_term + i_term + d_term
        return u


class Graph(object):
    """
    A graph for storing spatial information, handles the selection of new random waypoints and the path-finding to go
    from one waypoint to another
    """
    def __init__(self):
        self.nodes = list()
        self.idx_counter = 0
        self.minx, self.maxx = -55, 55
        self.miny, self.maxy = -55, 55
        self.nsteps = 2000
        self.xstep = (self.maxx - self.minx) / (self.nsteps - 1)
        self.ystep = (self.maxy - self.miny) / (self.nsteps - 1)
        self.xvals = np.linspace(-55, 55, 2000)
        self.yvals = np.linspace(-55, 55, 2000)
        self.x, self.y = np.meshgrid(self.xvals, self.yvals)
        self.outer_radius = 4
        self.inner_radius = 3.5
        self.pdf = np.zeros_like(self.x)

    def annulus(self, mu, x, y):
        xmini, xmaxi, ymini, ymaxi = self.get_xy_idxs(mu[0], mu[1], self.outer_radius + .1)
        subx = x[ymini:ymaxi, xmini:xmaxi] - mu[0]
        suby = y[ymini:ymaxi, xmini:xmaxi] - mu[1]
        dist = np.sqrt(subx * subx + suby * suby)
        subf = np.zeros_like(dist)
        subf[dist < self.outer_radius] = 1
        subf[dist < self.inner_radius] = -np.inf
        return subf, xmini, xmaxi, ymini, ymaxi

    def get_suppresion(self, mu, x, y):
        xmini, xmaxi, ymini, ymaxi = self.get_xy_idxs(mu[0], mu[1], self.outer_radius + .1)
        subx = x[ymini:ymaxi, xmini:xmaxi] - mu[0]
        suby = y[ymini:ymaxi, xmini:xmaxi] - mu[1]
        dist = np.sqrt(subx * subx + suby * suby)
        subf = np.zeros_like(dist)
        subf[dist < self.inner_radius] = -np.inf
        return subf, xmini, xmaxi, ymini, ymaxi

    def get_kill(self, mu, x, y):
        xmini, xmaxi, ymini, ymaxi = self.get_xy_idxs(mu[0], mu[1], self.outer_radius + .1)
        subx = x[ymini:ymaxi, xmini:xmaxi] - mu[0]
        suby = y[ymini:ymaxi, xmini:xmaxi] - mu[1]
        dist = np.sqrt(subx * subx + suby * suby)
        subf = np.zeros_like(dist)
        subf[dist < self.outer_radius] = -np.inf
        return subf, xmini, xmaxi, ymini, ymaxi

    def get_xy_idxs(self, xc, yc, radius):
        xmin_idx = np.max([np.argmin(np.abs((xc - radius) - self.xvals)) - 1, 0])
        xmax_idx = np.min([np.argmin(np.abs((xc + radius) - self.xvals)) + 1, len(self.yvals) - 1])
        ymin_idx = np.max([np.argmin(np.abs((yc - radius) - self.yvals)) - 1, 0])
        ymax_idx = np.min([np.argmin(np.abs((yc + radius) - self.yvals)) + 1, len(self.yvals) - 1])
        return int(xmin_idx), int(xmax_idx), int(ymin_idx), int(ymax_idx)

    def get_disk(self, mu, x, y):
        xmini, xmaxi, ymini, ymaxi = self.get_xy_idxs(mu[0], mu[1], self.outer_radius + .1)
        subx = x[ymini:ymaxi, xmini:xmaxi] - mu[0]
        suby = y[ymini:ymaxi, xmini:xmaxi] - mu[1]
        dist = np.sqrt(subx * subx + suby * suby)
        subf = np.zeros_like(dist)
        subf[dist < self.outer_radius] = 1
        return subf, xmini, xmaxi, ymini, ymaxi

    def find_potential_links(self):
        for node1 in self.nodes:
            for node2 in self.nodes:
                dist = np.sqrt(np.sum((node1.pos - node2.pos) ** 2))
                if dist < (self.outer_radius * 1.2) and node2 not in node1.neighbors.keys():
                    if not (node1.reached == -1 or node2.reached == -1):
                        node1.connect_to(node2, 0)
                if dist < (self.outer_radius * 1.2):
                    if node1.reached == -1 or node2.reached == -1:
                        node1.connect_to(node2, -1)

    def get_distribution(self):
        pdf = copy.copy(self.pdf)
        pdf[pdf == -np.inf] = 0
        if not (pdf == 0).all():
            pdf = pdf - np.max(pdf) + 1
        pdf[pdf < 0] = 0
        return pdf

    def local_distribution(self, node):
        local, xmini, xmaxi, ymini, ymaxi = self.get_disk(node.pos, self.x, self.y)
        local_pdf = copy.copy(self.pdf[ymini:ymaxi, xmini:xmaxi])
        local_pdf[local_pdf == -np.inf] = 0
        local_pdf = local_pdf * local
        if not (local_pdf == 0).all():
            local_pdf = local_pdf - np.max(local_pdf) + 1
        local_pdf[local_pdf < 0] = 0
        subx = self.x[ymini:ymaxi, xmini:xmaxi]
        suby = self.y[ymini:ymaxi, xmini:xmaxi]
        return local_pdf, subx, suby

    def plot_graph(self):
        # plt.clf()
        plt.ion()
        for node in self.nodes:
            node.plot_edges()
        for node in self.nodes:
            node.plot_node()

        plt.xlim([-55, 55])
        plt.ylim([-55, 55])
        plt.gca().set_aspect('equal')

        # xlims = plt.xlim()
        # ylims = plt.ylim()
        # xlims = [xlims[0]-self.outer_radius-1, xlims[1]+self.outer_radius+1]
        # ylims = [ylims[0]-self.outer_radius-1, ylims[1]+self.outer_radius+1]
        # plt.imshow(1-self.pdf, origin='lower', extent=[-55, 55, -55, 55], vmin=-1, vmax=1)
        # plt.xlim(xlims)
        # plt.ylim(ylims)
        plt.pause(0.01)

    def _check_pos_valid(self, pos):
        if pos is None:
            return False
        valid = True
        for node in self.nodes:
            valid = valid and node.check_not_close(pos)
        return valid

    def get_rand_pos_from_node(self, node):
        if len(node.neighbors) >= 7:
            node.set_enclosed(True)
            return None
        else:
            pdf, x, y = self.local_distribution(node)
            if (pdf == 0).all():
                node.set_enclosed(True)
                print('get_rand_pos_from_node:dist=0')
                return None
            else:
                rand_pos = self.sample(pdf, x, y)
                return rand_pos

    def calc_node_enclosed(self, node):
        if not node.enclosed:
            if len(node.neighbors) >= 7:
                node.set_enclosed(True)
            else:
                pdf, x, y = self.local_distribution(node)
                if (pdf == 0).all():
                    node.set_enclosed(True)

    def calc_enclosed(self):
        for node in self.nodes:
            self.calc_node_enclosed(node)

    def get_rand_node(self):
        print('# Nodes:{}'.format(len(self.nodes)))
        self.calc_enclosed()
        rand_node = random.choice(self.nodes)
        while len(rand_node.neighbors.keys()) == 0 or rand_node.reached == -1 or rand_node.enclosed:
            rand_node = random.choice(self.nodes)
        return rand_node

    def sample(self, pdf, x, y):
        x_idxs, y_idxs = np.where(pdf == 1)
        # while len(x_idxs) == 0:
        #     x_idxs, y_idxs = np.where(np.random.rand(*pdf.shape) < pdf)
        i = np.random.choice([i for i in range(len(x_idxs))])
        x_idx = x_idxs[i]
        y_idx = y_idxs[i]
        x_val = x[0, y_idx]
        y_val = y[x_idx, 0]
        return np.array([x_val, y_val])

    def get_near_rand_pos(self, pos):
        viable_nodes = list()
        dists = list()
        weights = list()
        print('get_near_rand_pos')
        for n in self.nodes:
            if not n.enclosed:
                viable_nodes.append(n)
                dist = np.sqrt(np.sum((pos - n.pos) ** 2))
                dists.append(dist)
                weight = 1/(1+np.exp(dist-2))
                weights.append(weight)
        rand_node = random.choices(viable_nodes, np.exp(np.array(weights)), k=1)[0]
        # weights.sort()
        # print(weights)
        rand_pos = self.get_rand_pos_from_node(rand_node)
        return rand_pos, rand_node

    # if this function returns None, it probably means there is no more space to explore
    def get_rand_pos(self):
        pdf = self.get_distribution()
        if (pdf <= 0).all():
            return None, None
        rand_pos = self.sample(pdf, self.x, self.y)
        print('rand_pos:{}'.format(rand_pos))
        rand_node = self.get_nearest_node(rand_pos)
        return rand_pos, rand_node

    def get_nearest_node(self, pos):
        print('get_nearest_node')
        nearest_dist = np.inf
        nearest_node = None
        for node in self.nodes:
            dist = np.sqrt(np.sum((pos - node.pos) ** 2))
            if dist < nearest_dist and node.reached == 1:
                nearest_dist = dist
                nearest_node = node
        return nearest_node

    # def add_node_waypoint(self, start_node, waypoint2):
    #     distance = np.sqrt(np.sum((start_node.pos - waypoint2) ** 2))
    #     new_node = Node(waypoint2, self.idx_counter)
    #     self.idx_counter += 1
    #     start_node.add_neighbor(new_node, distance)
    #     new_node.add_neighbor(start_node, distance, 1)
    #     self.nodes.append(new_node)

    def add_node(self, node):
        if node not in self.nodes:
            # print('\tadding node...')
            node.idx = self.idx_counter
            self.idx_counter += 1
            self.nodes.append(node)
            self.find_potential_links()
            node.graph = self
            subf, xmini, xmaxi, ymini, ymaxi = self.get_suppresion(node.pos, self.x, self.y)
            self.pdf[ymini:ymaxi, xmini:xmaxi] += subf

    def connect_nodes(self, idx1, idx2):
        if isinstance(idx1, int):
            node1 = [node for node in self.nodes if node.idx == idx1][0]
            node2 = [node for node in self.nodes if node.idx == idx2][0]
        else:
            node1 = idx1
            node2 = idx2
        distance = np.sqrt(np.sum((node1.pos - node2.pos) ** 2))
        node1.add_neighbor(node2, distance)
        node2.add_neighbor(node1, distance)

    def clear_nodes(self):
        for node in self.nodes:
            node.distance = np.inf

    def traverse(self, start_node, end_node):
        self.find_potential_links()
        print('{} ==> {}'.format(start_node.idx, end_node.idx))
        # start_node = [node for node in self.nodes if node.idx == start_node][0]
        # end_node = [node for node in self.nodes if node.idx == end_node][0]
        # print('traversing: {} steps'.format(len(self.nodes) ** 2))
        start_node.distance = 0

        for i in range(len(self.nodes)):
            for node in self.nodes:
                if node.distance != np.inf:
                    node.traverse()
        path = [end_node]
        while path[-1] is not start_node:
            print('\t{}'.format(path[-1].idx))
            path.append(path[-1].route_back())
        path.reverse()
        self.clear_nodes()
        for i in range(len(path)-1):
            path[i]._plot_edge(path[i+1], True)
        self.plot_graph()
        return path


class Node(object):
    """
    Nodes in a graph correspond to locations in space
    """
    def __init__(self, pos, idx):
        self.pos = np.array(pos)
        self.idx = idx
        self.distance = np.inf
        self.neighbors = dict()
        self.fails = 0
        self.reached = 0
        self.enclosed = False
        self.graph = None
        self.plotted = False

    @staticmethod
    def get_distance(node1, node2):
        return np.sqrt(np.sum((node1.pos - node2.pos) ** 2))

    def set_enclosed(self, enclosed):
        # print('ENCLOSED')
        self.enclosed = enclosed
        self.plotted = False

    def set_reached(self, reached):
        self.reached = reached
        if self.reached == 1:
            subf, xmini, xmaxi, ymini, ymaxi = self.graph.annulus(self.pos, self.graph.x, self.graph.y)
            self.graph.pdf[ymini:ymaxi, xmini:xmaxi] += subf
            # self.graph.pdf += self.graph.annulus(self.pos, self.graph.x, self.graph.y)
        if self.reached == -1:
            subf, xmini, xmaxi, ymini, ymaxi = self.graph.get_kill(self.pos, self.graph.x, self.graph.y)
            self.graph.pdf[ymini:ymaxi, xmini:xmaxi] += subf

    def _plot_edge(self, node, path):
        x = [self.pos[0], node.pos[0]]
        y = [self.pos[1], node.pos[1]]
        linewidth = .5
        if path:
            plt.plot(x, y, color=[0, 1, 0], linewidth=.75)
        else:
            if self.neighbors[node]['status'] == 1:
                plt.plot(x, y, color=[0, 0, 0], linewidth=linewidth)
            elif self.neighbors[node]['status'] == 0:
                plt.plot(x, y, color=[.7, .7, .7], linewidth=linewidth)
            else:
                plt.plot(x, y, color=[1, 0, 0], linewidth=linewidth)

    def plot_edges(self):
        if not self.plotted:
            for node in self.neighbors.keys():
                if self.neighbors[node]['status'] == 0:
                    self._plot_edge(node, False)
            for node in self.neighbors.keys():
                if self.neighbors[node]['status'] == -1:
                    self._plot_edge(node, False)
            for node in self.neighbors.keys():
                if self.neighbors[node]['status'] == 1:
                    self._plot_edge(node, False)

    def plot_node(self):
        if not self.plotted:
            if self.reached == -1:
                plt.plot(self.pos[0], self.pos[1], 'o', color=[1, 0, 0], markersize=1)
            else:
                if self.enclosed:
                    plt.plot(self.pos[0], self.pos[1], 'o', color=[0, 0, 1], markersize=1)
                else:
                    plt.plot(self.pos[0], self.pos[1], 'o', color=[0, 0, 0], markersize=1)
            self.plotted = True

    def _rand_pos(self):
        return self.pos + np.random.uniform([-self.graph.outer_radius, -self.graph.outer_radius], [self.graph.outer_radius, self.graph.outer_radius])

    def _check_pos_valid(self, apos):
        dist = np.sqrt(np.sum((apos - self.pos) ** 2))
        return self.graph.inner_radius < dist < self.graph.outer_radius

    def check_not_close(self, apos):
        dist = np.sqrt(np.sum((apos - self.pos) ** 2))
        return self.graph.inner_radius < dist

    def get_rand_pos(self):
        rand_pos = self._rand_pos()
        while not self._check_pos_valid(rand_pos):
            rand_pos = self._rand_pos()
        return rand_pos

    def set_status(self, node, status):
        if node in self.neighbors.keys():
            self.neighbors[node]['status'] = status
        if self in node.neighbors.keys():
            node.neighbors[self]['status'] = status
        self.plotted = False

    def connect_to(self, node, status):
        distance = Node.get_distance(self, node)
        self.add_neighbor(node, distance, status)
        node.add_neighbor(self, distance, status)

    def add_neighbor(self, node, distance, status):
        self.neighbors[node] = {
            'distance': distance,
            'status': status
        }
        if self.graph is not None:
            self.graph.calc_node_enclosed(self)
        self.plotted = False

    def traverse(self):
        for node in self.neighbors.keys():
            if node.distance > self.distance + self.neighbors[node]['distance'] and self.neighbors[node]['status'] != -1:   # we have a shorter path
                node.distance = self.distance + self.neighbors[node]['distance']

    def route_back(self):
        nearest_node = None
        nearest_distance = np.inf
        # print('[N:{}] route_back: len(neighbors) = {}'.format(self.idx, len(self.neighbors.keys())))
        for node in self.neighbors.keys():
            if node.distance < nearest_distance:
                nearest_distance = node.distance
                nearest_node = node
        if nearest_node is None:
            print(30 * '=')
            print('NODE:ROUTE_BACK error for [NODE {}]'.format(self.idx))
            print('NEIGHBORS: <')
            for node in self.neighbors.keys():
                print('\tnode {}'.format(node.idx))
            print('>')
        return nearest_node


class ForModule(nn.Module):
    """
    The Neural Network that esimates best next target positions in order to move in a certain direction
    """
    def __init__(self, layer_dims):
        super(ForModule, self).__init__()
        self.layer_dims = layer_dims
        layers = list()
        for i in range(len(layer_dims)-1):
            if i < len(layer_dims)-2:
                layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
                # layers.append(nn.BatchNorm1d(layer_dims[i+1]))
                layers.append(nn.PReLU())
            else:
                print('non-BatchNorm layer')
                layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                layers.append(nn.PReLU())

        self.layers = nn.ModuleList(layers)
        # this module maps <pose_t, pose_t+1, sense_t> to relative_disp_t+1

    def process(self, data_dict):
        x = extract_inputs(data_dict)
        for layer in self.layers:
            x = layer.process(x)
        data_dict['~disp_vect'] = x
        return data_dict

    def save_model(self, path, filename):
        model = {
            'model': ForModule,
            'state_dict': self.state_dict(),
            'layer_dims': self.layer_dims
        }
        torch.save(model, path + filename)

    @staticmethod
    def load_model(path, filename):
        checkpoint = torch.load(path + filename)
        model = checkpoint['model'](checkpoint['layer_dims'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


class Optimizer(object):
    """
    A class encapsulating the training code for the neural network
    """
    def __init__(self, model, dataset, lr):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.dataset = dataset

    def get_sample(self):
        minibatch = get_minibatch(self.dataset, 1)
        minibatch = self.model.process(minibatch)
        # weights = torch.sqrt(torch.sum(minibatch['disp_vect'] ** 2))
        output = minibatch['~disp_vect'][:, 0:2]
        loss = self.criterion(output, minibatch['disp_vect'])
        return minibatch, loss.detach().item(), self.get_relative_error_mag(minibatch)

    def step(self, batch_size):
        minibatch = get_minibatch(self.dataset, batch_size)
        minibatch = self.model.process(minibatch)
        output = minibatch['~disp_vect'][:, 0:2]
        loss = self.criterion(100*output, 100*minibatch['disp_vect'])/10000
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item(), self.get_relative_error_mag(minibatch)

    def train(self, iters, batch_size):
        rel_error_hist = np.zeros(iters)
        error_hist = np.zeros(iters)
        for i in range(iters):
            error_hist[i], rel_error_hist[i] = self.step(batch_size)
            if i % 10 == 0:
                print('\tLOG_ERR: {}, REL_ERR: {}'.format(np.log(error_hist[i]), rel_error_hist[i]))
        return error_hist, rel_error_hist

    def get_relative_error_mag(self, minibatch):
        output = minibatch['~disp_vect'][:, 0:2]
        disp_error = (minibatch['disp_vect'] - output).detach().numpy()
        error_mag = np.sqrt(np.sum(disp_error ** 2, 1))
        disp_mag = np.sqrt(np.sum(minibatch['disp_vect'].detach().numpy() ** 2, 1))
        return np.mean(error_mag / disp_mag)

