import homeokinesis as hk
import torch
import torch.nn as nn
import numpy as np


def quick_func_spec(name, inputs, outputs, state_spec):
    f_spec = {
        'network_name': name,
        'inputs': inputs,
        'outputs': outputs,
        'state_spec': state_spec,
        'layer_spec': {
            'n_layers': 3
        },
        'nonlin': nn.Tanh
    }
    return f_spec


def quick_adapt_spec(target, output, models):
    a_spec = {
        'models': models,
        'target': target,
        'output': output,
        'optimizer': torch.optim.SGD,
        'optimizer_args': {'lr': 0.1, 'weight_decay': False},
        'loss_module': nn.MSELoss
    }
    return a_spec


state_dict = {
    'x_t-2': torch.zeros(1, 10),
    'x_t-1': torch.zeros(1, 10)+1,
    'x_t': torch.zeros(1, 10)+2,
    'y_t-2': torch.zeros(1, 10)+3,
    'y_t-1': torch.zeros(1, 10)+4,
    'y_t': torch.zeros(1, 10)+5
}

shift_func = hk.get_time_shift_func(['x', 'y'], ['t-2', 't-1', 't'])
print(state_dict)
shift_func(state_dict)
print(state_dict)

exit()

state_spec = {
    'x1': 10,
    'x2': 10,
    'x3': 10,
    'y1': 10,
    'y2': 10,
    'y3': 10,
    '~z1': 10,
    '~z2': 10,
    '~z3': 10,
    'z1': 10,
    'z2': 10,
    'z3': 10
}
state_dict = hk.state_dict_from_spec(state_spec, 'zeros')

model_specs = [
    quick_func_spec('f4', ['~z1', 'y3'], ['~z2', '~z3'], state_spec),
    quick_func_spec('f2', ['~z3'], ['y2', 'y3'], state_spec),
    quick_func_spec('f1', ['x1', 'x2'], ['y1'], state_spec),
    quick_func_spec('f3', ['x1', 'y2'], ['~z1'], state_spec),
]

ordered_funcs = hk.order_funcs(state_spec, model_specs)
print(ordered_funcs)

# brain = hk.Brain(state_dict)
# brain.add_networks(model_specs)
# brain.add_adapters(adapter_specs)

# for i in range(1000):
#     x = torch.rand(1, 10)
#     y = torch.tensor(x)
#
#     brain.state_dict['x'] = x
#     brain.state_dict['y'] = y
#
#     brain.forward()
#     brain.adapt()
#
# x = torch.rand(1, 10)
# brain.state_dict['x'] = x
# brain.forward()
#
# print('y  : {}'.format(x))
# print('y1 : {}'.format(brain.state_dict['y1']))
# print('y2 : {}'.format(brain.state_dict['y2']))
