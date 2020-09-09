import json
from pathlib import Path
from typing import Union, Dict

import torch
import numpy as np

from logger import get_logger

class MetricTracker(object):
    """
    A simple class ment for storing and saving training loss history
    and validation metric history in json file
    """

    TRAINING_KEY = 'training'
    VALIDATION_KEY = 'validation'
    CONFIG_KEY = 'config'

    def __init__(self,
                 config: dict):
        """
        Args:
            config (dict): The config dict which initiates the network
        """

        self.logger = get_logger(name=__name__)

        self.results = dict()
        self.results[self.TRAINING_KEY] = dict()
        self.results[self.VALIDATION_KEY] = dict()
        self.results[self.CONFIG_KEY] = dict()

    def resume(self, resume_path: Union[str, Path]):
        """
        Resumes MetricTracker from previous state
        NB! Overwrites anything stored except the config dict
        Args:
            resume_path (str, pathlib.Path): The previous saved MetricTracker object
        """
        if not isinstance(resume_path, (str, Path)):
            TypeError('resume_path is not of type str or Path but {}'.format(type(resume_path)))

        if not Path(resume_path).is_file():
            self.logger.warning('{} is not a file will not resume\
                                 from MetricTracker instance.'.format(str(resume_path)))

        with open(str(resume_path), 'r') as inifile:
            prev = json.loads(inifile)

        if self.TRAINING_KEY not in prev.keys() or self.VALIDATION_KEY not in prev.keys():
            self.logger.warning('The given file does not have the training or validation key,\
                                 will not resume from prior checkpoint.')
            return

        if self.CONFIG_KEY in prev.keys():
            if prev[self.CONFIG_KEY] != self.results[self.CONFIG_KEY]:
                self.logger.warning('Non identical configs found,\
                                     this instance will store the new config.')

        self.results[self.TRAINING_KEY].update(prev[self.TRAINING_KEY])
        self.results[self.VALIDATION_KEY].update(prev[self.VALIDATION_KEY])


    def training_update(self,
                        loss: Dict[str, list],
                        epoch: int):
        """
        Appends new training history
        Args:
            loss (list, np.ndarray, torch.Tensor): The loss history for this batch
            epoch (int): The epoch or iteration number, repeated numbers will overwrite
                         previous history
        """

        epoch = 'epoch_' + str(epoch)
        self.results[self.TRAINING_KEY][epoch] = loss

    def validation_update(self,
                          metrics: Dict[str, list],
                          epoch: int):
        """
        Appends new validation history
        Args:
            metrics (dict): A dict matching the metric to the score for one/multiple metrics
            epoch (int): The epoch or iteration number, repeated numbers will overwrite
                         previous history
        """

        epoch = 'epoch_' + str(epoch)
        self.results[self.VALIDATION_KEY][epoch] = metrics

    def write_to_file(self, path: Union[str, Path]):
        """
        Writes MetricTracker to file
        Args:
            path (str, pathlib.Path): Path where the file is stored,
                                      remember to have .json suffix
        """
        Path(path).mkdir(parents=True)  # Missing parents are quite the letdown
        path = str(path)

        with open(path, 'w') as outfile:
            json.dump(self.results, outfile, indent=4)

