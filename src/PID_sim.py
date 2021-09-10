import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
import multiprocessing as mp
import os
from utils import get_path
import numpy as np
import utils
import ant_control
from mujoco_py import load_model_from_path, MjSim


def gen_t_params(T, dt):
    return {
        'T': T,
        'dt': dt,
        't_series': np.arange(0, T, dt)
    }


def gen_time_series(k_params, t_params, target):
    model = load_model_from_path(os.path.join(get_path(2), 'xml', 'ant.xml'))
    sim = MjSim(model)

    qpos_series = np.zeros_like(target)
    t_series = t_params['t_series']
    dt = t_params['dt']

    P = k_params[0]
    I = k_params[1]
    D = k_params[2]

    pid = ant_control.PID(8, P, I, D, dt, 0.01, 0.01, 10)

    for i in range(0, t_series.size):
        # print('{}/{}'.format(i, t_series.size))
        pid.set_target(target[i, :])

        sim.data.qacc[:] = np.clip(sim.data.qacc[:], -14000, 14000)
        qpos = sim.data.qpos[7:]
        qpos_series[i, :] = qpos
        ctrl, success = pid.forward(qpos)
        sim.data.ctrl[:] = ctrl
        sim.step()

    return {
        't': t_series,
        'x': qpos_series,
        'target': target
    }


def gen_control_data_parallel(job):
    k_params = job['k_params']
    t_params = job['t_params']
    num_trials = job['num_trials']
    trials = list()
    for i in range(num_trials):
        target = generate_target(t_params)
        trials.append(gen_time_series(k_params, t_params, target))

    MSE = calculate_score(trials, t_params)
    data = {
        'trials': trials,
        'MSE': MSE
    }
    return data


def squared_derivative_error_integral(target, signal, t_params):
    d_target = (target[1:] - target[:-1]) / t_params['dt']
    d_signal = (signal[1:] - signal[:-1]) / t_params['dt']
    return np.sum((d_target - d_signal) ** 2) * t_params['dt'] / t_params['T']


def squared_error_integral(target, signal, t_params):
    return np.sum((target - signal) ** 2) * t_params['dt'] / t_params['T']


def calculate_score(trials, t_params):
    SEI_list = list()
    for trial in trials:
        SEI = squared_error_integral(trial['target'], trial['x'], t_params)
        # SDEI = squared_derivative_error_integral(trial['target'], trial['x'][0], t_params)
        SEI_list.append(SEI / 30)
    MSE = np.mean(np.array(SEI_list))
    return MSE


def generate_target(t_params):
    t_series = t_params['t_series']
    target = np.zeros((t_series.size, 8))
    min_theta = np.array([-30, 30, -30, -70, -30, -70, -30, 30])
    max_theta = np.array([30, 70, 30, -30, 30, -30, 30, 70])
    vect = np.random.uniform(min_theta * np.pi / 180, max_theta * np.pi / 180)

    for i in range(t_series.size):
        if i % 500 == 0:
            vect = np.random.uniform(min_theta * np.pi / 180, max_theta * np.pi / 180)
        target[i, :] = vect

    return target


def mutate_k_params(k_params):
    mutant_params = [p for p in k_params]
    choices = choice([0, 1, 2], len(k_params), [.1, .45, .45])
    for i in range(len(k_params)):
        if choices[i] == 0:
            pass
        if choices[i] == 1:
            mutant_params[i] += np.random.normal(0, .5)
        if choices[i] == 2:
            mutant_params[i] *= np.random.uniform(0.5, 1.5)
        if mutant_params[i] < 0:
            mutant_params[i] = 0
    return mutant_params
    # return [round(p, 4) for p in mutant_params]

# k_params = [2.0980263160608583, 0.833227797012427, 0.0]
# t_params = gen_t_params(60, 0.001)
# target = generate_target(t_params)
# results = gen_time_series(k_params, t_params, target)
# SEI = squared_error_integral(results['target'], results['x'], t_params)
# print('SEI: {}'.format(SEI))
# plt.plot(results['t'], results['target'][:, 0])
# plt.plot(results['t'], results['x'][:, 0])
# plt.show()
# exit()
#
# t_params = gen_t_params(60, 0.001)
# k_params = [2.1, 0.833, 0.0]
#
# dp = 0.01
#
# for i in range(100):
#     target = generate_target(t_params)
#     # baseline
#     results = gen_time_series(k_params, t_params, target)
#     E = squared_error_integral(results['target'], results['x'], t_params)
#     print('error: {}'.format(E))
#     # perturbing P
#     dP_k_params = [k for k in k_params]
#     dP_k_params[0] += dp
#     dP_results = gen_time_series(dP_k_params, t_params, target)
#     E_P = squared_error_integral(dP_results['target'], dP_results['x'], t_params)
#     # perturbing I
#     dI_k_params = [k for k in k_params]
#     dI_k_params[1] += dp
#     dI_results = gen_time_series(dI_k_params, t_params, target)
#     E_I = squared_error_integral(dI_results['target'], dI_results['x'], t_params)
#     # pertubring D
#     dD_k_params = [k for k in k_params]
#     dD_k_params[2] += dp
#     dD_results = gen_time_series(dD_k_params, t_params, target)
#     E_D = squared_error_integral(dD_results['target'], dD_results['x'], t_params)
#
#     dE_dP = (E_P - E) / dp
#     dE_dI = (E_I - E) / dp
#     dE_dD = (E_D - E) / dp
#
#     error_vect = [dE_dP, dE_dI, dE_dD]
#     new_k_params = [k_params[i] - error_vect[i] for i in range(3)]
#     k_params = new_k_params
#
# print('-----------------------')
# print(k_params)
#
# exit()

if __name__ == '__main__':
    t_params = gen_t_params(1, 0.001)
    t_series = t_params['t_series']

    mode = 'testing'

    if mode == 'training':

        with mp.Pool(processes=4) as pool:

            num_trials = 1
            k = 5
            r = 10
            generations = 1000
            sensor_noise = 0.001

            # generate initial 'winning' population
            k_params_1st = [1, 2, .1]
            k_params_2nd = [1, 1, 1]
            population = [k_params_1st, k_params_2nd]

            # for i in range(4):
            #     population.append(mutate_k_params(k_params_1st))
            #     population.append(mutate_k_params(k_params_2nd))

            for generation in range(generations):
                t_params = gen_t_params(60, 0.001)
                t_series = t_params['t_series']

                # create new population
                print('pop size: {}'.format(len(population)))
                for i in range(len(population)):
                    for j in range(r):
                        population.append(mutate_k_params(population[i]))
                print('new pop size: {}'.format(len(population)))

                # test the population
                jobs = [{
                    'k_params': p,
                    't_params': t_params,
                    'num_trials': num_trials
                } for p in population]

                pop_trials = pool.map(gen_control_data_parallel, jobs)
                MSE_list = [trials['MSE'] for trials in pop_trials]

                # pick the best k solutions
                idxs = np.argsort(MSE_list)
                population = [population[idx] for idx in idxs[0:(k-1)]]

                print('{} : {}, {}'.format(generation, MSE_list[idxs[0]], population[0]))
            k_params_1st = population[0]
            job = {
                'k_params': k_params_1st,
                't_params': t_params,
                'num_trials': 1
            }
            trials = gen_control_data_parallel(job)
            for trial in trials['trials']:
                plt.plot(t_params['t_series'], trial['x'][0], '-', color=[1, 0, 0])
                plt.plot(t_params['t_series'], trial['target'], '-', color=[0, 0, 1])
            # MSE = calculate_score(trials, target, t_params)
            # plt.title(MSE)
            plt.show()

    else:
        t_params = gen_t_params(6, 0.001)
        target = generate_target(t_params)
        k_params_1 = [60, 40, 1]
        k_params_2 = [70, 30, 1]

        results_1 = gen_time_series(k_params_1, t_params, target)
        results_2 = gen_time_series(k_params_2, t_params, target)
        plt.plot(results_1['t'], target[:, 0], '-', color=[0, 0, 0])
        plt.plot(results_1['t'], results_1['x'][:, 0], '-', color=[1, 0, 0])
        plt.plot(results_2['t'], results_2['x'][:, 0], '-', color=[0, 1, 0])
        plt.ylim([-1, 1])
        # plt.title(trials['MSE'])
        # MSE = calculate_score(trials, target, t_params)

        plt.show()

