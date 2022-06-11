import datetime
import json
import os

import torch
import shutil
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class TBLogger:
    def __init__(self, args, exp_label):
        self.output_name = exp_label + '_' + \
            str(args.seed) + '_' + \
            datetime.datetime.now().strftime('_%m_%d_%H_%M_%S')
        try:
            log_dir = args.results_log_dir
        except AttributeError:
            log_dir = args['results_log_dir']

        if log_dir is None:
            dir_path = os.path.abspath(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), os.pardir))
            dir_path = os.path.join(dir_path, 'logs')
        else:
            dir_path = log_dir

        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except:
                dir_path_head, dir_path_tail = os.path.split(dir_path)
                if len(dir_path_tail) == 0:
                    dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                os.mkdir(dir_path_head)
                os.mkdir(dir_path)

        try:
            self.full_output_folder = os.path.join(os.path.join(dir_path, 'logs_{}'.format(args.env_name)),
                                                   self.output_name)
        except:
            self.full_output_folder = os.path.join(os.path.join(dir_path, 'logs_{}'.format(args["env_name"])),
                                                   self.output_name)

        self.writer = SummaryWriter(log_dir=self.full_output_folder)

        print('logging under', self.full_output_folder)

        if not os.path.exists(self.full_output_folder):
            os.makedirs(self.full_output_folder)

        # copy all the files in current working directory to the output folder, excpet for log folder and environments/mujoco
        shutil.copytree(os.getcwd(), self.full_output_folder +
                        '/source_code', ignore=shutil.ignore_patterns('logs', 'tmp_results', '.git', '__pycache__'))

        with open(os.path.join(self.full_output_folder, 'config.json'), 'w') as f:
            try:
                config = {k: v for (k, v) in vars(
                    args).items() if k != 'device'}
            except:
                config = args
            config.update(device=device.type)
            json.dump(config, f, indent=2)

        self.update_every = 5
        self.last_update = 0
        self.record = {}
        self.step = {}

    def add(self, name, value, x_pos, force_plot=False):
        self.writer.add_scalar(name, value, x_pos)
        if type(value) is torch.Tensor:
            value = value.detach().cpu()

        if name in self.record.keys():
            self.record[name].append(value)
            self.step[name].append(x_pos)
        else:
            self.record[name] = [value]
            self.step[name] = [x_pos]

        self.last_update += 1
        if (self.last_update >= self.update_every) or force_plot:
            self.last_update = 0
            self.plot()

    def plot(self):
        for k, v in self.record.items():
            plt.figure()
            plt.plot(self.step[k], v)
            plt.title(k)
            # plt.savefig(os.path.join(self.full_output_folder, k + '.png'))
            plt.savefig('{}/{}.png'.format(self.full_output_folder, k))
            plt.close('all')
