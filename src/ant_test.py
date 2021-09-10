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


def set_target_position(mjsim, target):
    if target is None:
        mjsim.data.qpos[15] = 1000
        mjsim.data.qpos[16] = 1000
    else:
        mjsim.data.qpos[15] = target.pos[0]
        mjsim.data.qpos[16] = target.pos[1]
    mjsim.data.qpos[17] = 1


mode = 'training'
render = True
load_memory = True
write_memory = False
save_data = False


files = os.listdir(os.path.join(get_path(2), 'memory'))
files = [file for file in files if '.pkl' in file]
files_nums = [float(file[7:-4].replace('-', '.')) for file in files]
file_idx = np.argmax(files_nums)
old_memory_file = os.path.join(get_path(2), 'memory', files[file_idx])

dt = 0.001
model = load_model_from_path(os.path.join(get_path(2), 'xml', 'ant.xml'))
sim = MjSim(model)
print(sim.data.qpos.size)
# print(sim.data.xpos)
# exit()
viewer = MjViewer(sim)
sim_state = sim.get_state()

if mode == 'training':
    time_args = {
        'sim_time': '1000m',
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

if load_memory:
    controller.seq_controller.load_seq_mem(old_memory_file)

timestamp = utils.get_time_stamp()
pid_data_file = os.path.join('/Volumes/ALEX_SSD/HNSC_data', 'pid_{}.txt'.format(timestamp))
target_data_file = os.path.join('/Volumes/ALEX_SSD/HNSC_data', 'target_{}.txt'.format(timestamp))
new_memory_file = os.path.join(get_path(2), 'memory', 'memory_{}.pkl'.format(timestamp))

start_time = time.time()
timer = time.time()
buffer = ''
pos_val = 0
# with open(pid_data_file, 'w') as pid_f, open(target_data_file, 'w') as target_f:
try:
    sim.set_state(sim_state)
    t = 0
    controller.sense(parse_sensor_data(sim))
    controller.initialize()
    while t < total_sim_steps:  # inner simulation loop
        t += 1

        if time.time() - timer > 10:
            timer = time.time()
            print('{}%'.format(round(100 * t / total_sim_steps, 4)))

        sim.data.qacc[:] = np.clip(sim.data.qacc[:], -14000, 14000)
        set_target_position(sim, controller.pth_controller.next_waypoint)
        # pos_val += 0.00000001
        controller.sense(parse_sensor_data(sim))
        ctrl = controller.process()

        if save_data:
            save_data_dict = controller.get_save_data()
            save_data_txt = '{}\n'.format(utils.dict_to_b64(save_data_dict))
            buffer += save_data_txt

            # if controller.pid_controller.success == 1:
            #     target_f.write(save_data_txt)
            #     if mode == 'training':
            #         pid_f.write(buffer)
            #         buffer = ''

        sim.data.ctrl[:] = ctrl

        sim.step()

        # print(controller.sequence['name'])
        if render:
            if t < 2000:
                viewer.render()
            else:
                if t % 500 == 0:
                    viewer.render()

        if os.getenv('TESTING') is not None:
            break
except Exception as e:
    track = traceback.format_exc()
    print(track)

if write_memory:
    with open(new_memory_file, 'wb') as f:
        pickle.dump(controller.seq_controller.seq_mem_dict, f)
print('\nSimulation finished.')
print('Finished in {} seconds'.format(time.time() - start_time))
