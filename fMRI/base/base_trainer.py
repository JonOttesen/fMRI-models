from typing import List, Callable, Union, Dict
from abc import abstractmethod
from pathlib import Path
# from logger import TensorboardWriter

import torch
import numpy as np

from ..logger import get_logger
from .metrics import MetricTracker


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self,
                 model: torch.nn.Module,
                 loss_function: Callable,
                 metric_ftns: Dict[str, Callable],
                 optimizer: torch.optim,
                 config: dict,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 seed: int = None):

        # Reproducibility is a good thing
        if isinstance(seed, int):
            torch.manual_seed(seed)

        self.config = config
        self.logger = get_logger(name=__name__)

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self.prepare_device(config['n_gpu'])

        self.model = model.to(self.device)
        self.lr_scheduler = lr_scheduler

        # TODO: Use DistributedDataParallel instead
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.loss_function = loss_function
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        trainer_cfg = config['trainer']
        self.epochs = trainer_cfg['epochs']
        self.save_period = trainer_cfg['save_period']


        self.start_epoch = 1

        self.checkpoint_dir = trainer_cfg['save_dir']

        self.metric = MetricTracker(config=config)

        # setup visualization writer instance
        # self.writer = TensorboardWriter(config.log_dir, self.logger, trainer_cfg['tensorboard'])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Validation logic after an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            loss_dict = self._train_epoch(epoch)
            val_dict = self._valid_epoch(epoch)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # save logged information regarding this epoch/iteration
            self.metric.training_update(loss=loss_dict, epoch=epoch)
            self.metric.validation_update(metrics=val_dict, epoch=epoch)

            self.logger.info('Epoch/iteration {} completed, run mean statistics:'.format(epoch))
            # print logged informations to the screen
            for key, values in log.items():
                value = np.mean(np.array(values))
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            if epoch % self.save_period == 0:
                self.save_checkpoint(epoch)

    def prepare_device(self, n_gpu_use: int):
        """
        setup GPU device if available, move model into configured device
        Args:
            n_gpu_use (int): Number of GPU's to use
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def save_checkpoint(self, epoch):
        """
        Saving checkpoints at the given moment
        Args:
            epoch (int), the current epoch of the training
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
            }

        save_path = self.checkpoint_dir / 'epoch_' + str(epoch)
        filename = str(save_path / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def resume_checkpoint(self, resume_path: Union[str, Path]):
        """
        Resume from saved checkpoints
        Args:
            resume_path (str, pathlib.Path): Checkpoint path, either absolute or relative
        """
        if not isinstance(resume_path, (str, Path)):
            self.logger.warning('resume_path is not str or Path object but of type {},\
                                 aborting previous checkpoint loading'.format(type(resume_path)))
            return

        if not Path(resume_path).is_file():
            self.logger.warning('resume_path object does not exist, ensure that {} is correct,\
                                 aborting previous checkpoint loading'.format(str(resume_path)))
            return

        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))

        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning('Warning: Different architecture from that given in the config,\
                                 this may yield and exception while state_dict is loaded.')

        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Different optimizer from that given in the config,\
                                 optimizer parameters are not resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
