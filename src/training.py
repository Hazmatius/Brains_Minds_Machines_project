import numpy as np
import ant_control
import pickle
from datetime import datetime


def get_time_stamp():
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return timestamp


model_dir = '/HNSC/intuition/'
data_path = '/HNSC/data/dataset.pkl'
with open(data_path, 'rb') as f:
    dataset = pickle.load(f)

model = ant_control.ForModule([45, 100, 100, 100, 100, 100, 100, 100, 2])
# model = ant_control.ForModule.load_model(model_dir, '[-5.0]20210825-110550')

lr = 0.01
optim = ant_control.Optimizer(model, dataset, lr)
REL_ERR = 3
log_error = -4
while REL_ERR > 0.1:
    model.train()
    error_hist, rel_error_hist = optim.train(100, 90000)
    REL_ERR = rel_error_hist[-1]
    final_log_error = np.log(error_hist[-1])
    if final_log_error < log_error:
        model_name = '[{}]'.format(log_error) + get_time_stamp()
        model.save_model(model_dir, model_name)
        log_error = np.floor(final_log_error)
    if error_hist[-1] > error_hist[0]:
        lr = lr / 2
        optim = ant_control.Optimizer(model, dataset, lr)
    print('LR: {}'.format(lr))
print('done')