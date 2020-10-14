import json

from typing import Union
from pathlib import Path

import torch




class ConfigReader(object):

    def __init__(self, config: Union[dict, Union[str, Path]]):
        if isinstance(config, dict):
            self.config = config
        else:
            if not isinstance(config, (str, Path)):
                raise TypeError('Input must be of type dict or str/pathlib.Path not {}'.format(type(config)))
            with open(config, 'r') as inifile:
                self.config = json.load(inifile)

    def __getitem__(self, key):
        return self.config[key]

    def __str__(self):
        return self.config

    @property
    def shuffle(self):
        return self['data_loader']['args']['shuffle']

    @property
    def batch_size(self):
        return self['data_loader']['args']['batch_size']

    @property
    def num_workers(self):
        return self['data_loader']['args']['num_workers']

    def optimizer(self, model_params):
        optim = self.config['optimizer']
        optimizer = getattr(torch.optim, optim['type'])
        optimizer = optimizer(model_params,
                              lr=optim['args']['lr'],
                              weight_decay=optim['args']['weight_decay'],
                              amsgrad=optim['args']['amsgrad'])
        return optimizer

    def lr_scheduler(self, optimizer):
        lr_sched = self.config['lr_scheduler']
        lr_scheduler = getattr(torch.optim.lr_scheduler, lr_sched['type'])
        lr_scheduler = lr_scheduler(optimizer=optimizer,
                                    step_size=lr_sched['args']['step_size'],
                                    gamma=lr_sched['args']['gamma'])
        return lr_scheduler

    def configs(self):
        return self.config


