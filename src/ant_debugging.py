import os
from utils import get_path
import numpy as np
import utils
import time
from mujoco_py import load_model_from_path, MjSim, MjViewer
import traceback
import pickle
from ant_control import SYM
from ant_control import parse_sensor_data


mode = 'testing'
render = True
move_fast = False
load_memory = False
write_memory = False
save_data = False
waypoint_control = False

waypoint_seq = [np.random]
waypoint_seq = [
    np.array([0, 0]),
    np.array([20, 0]),
    np.array([0, 20]),
    np.array([20, 20])
]

old_memory_file = os.path.join(get_path(2), 'memory', 'memory_20210816-185933.pkl')
dt = 0.001
model = load_model_from_path(os.path.join(get_path(2), 'xml', 'ant.xml'))
sim = MjSim(model)
viewer = MjViewer(sim)
sim_state = sim.get_state()

if mode == 'training':
    time_args = {
        'sim_time': '100m',
        'dt': dt
    }
elif mode == 'testing':
    time_args = {
        'sim_time': '5m',
        'dt': dt
    }
else:
    raise Exception('mode must be either "training" or "testing"')
utils.parse_time(time_args)
total_sim_steps = time_args['n']
print('Running for {} timesteps'.format(total_sim_steps))

controller = SYM(dt, sim)
controller.waypoint_control = waypoint_control
controller.waypoint_seq = waypoint_seq
controller.move_fast = move_fast
# controller.pid.target = controller.generate_default_sequence()['sequence'][0]

if load_memory:
    with open(old_memory_file, 'rb') as f:
        controller.seq_mem_dict = pickle.load(f)
        controller.find_fastest_speeds()
        controller.counter = controller.get_max_seq_idx() + 1
        controller.sequence = controller.get_fastest_sequence(0)

timestamp = utils.get_time_stamp()
pid_data_file = os.path.join('/Volumes/ALEX_SSD/HNSC_data', 'pid_{}.txt'.format(timestamp))
target_data_file = os.path.join('/Volumes/ALEX_SSD/HNSC_data', 'target_{}.txt'.format(timestamp))
new_memory_file = os.path.join(get_path(2), 'memory', 'memory_{}.pkl'.format(timestamp))

start_time = time.time()
timer = time.time()
# with open(pid_data_file, 'w') as pid_f, open(target_data_file, 'w') as target_f:
try:
    sim.set_state(sim_state)
    t = 0
    while t < total_sim_steps:  # inner simulation loop
        t += 1

        if time.time() - timer > 10:
            timer = time.time()
            print('{}%'.format(round(100 * t / total_sim_steps, 4)))

        sim.data.qacc[:] = np.clip(sim.data.qacc[:], -14000, 14000)
        controller.sense(parse_sensor_data(sim))
        ctrl = controller.forward()
        # print(controller.sequence['name'])

        # if save_data:
        #     if controller.success == 1:
        #         save_data_dict = controller.get_save_data()
        #         save_data_txt = utils.dict_to_b64(save_data_dict)
        #         target_f.write('{}\n'.format(save_data_txt))
        #
        #     if mode == 'training':
        #         save_data_dict = controller.get_save_data()
        #         save_data_txt = utils.dict_to_b64(save_data_dict)
        #         pid_f.write('{}\n'.format(save_data_txt))

        sim.data.ctrl[:] = ctrl

        sim.step()

        viz = controller.viz

        # print(controller.sequence['name'])
        if render:
            if t % 1 == 0:
                viewer.render()

        if os.getenv('TESTING') is not None:
            break
except Exception as e:
    track = traceback.format_exc()
    print(track)

if write_memory:
    with open(new_memory_file, 'wb') as f:
        pickle.dump(controller.seq_mem_dict, f)
print('\nSimulation finished.')
print('Finished in {} seconds'.format(time.time() - start_time))