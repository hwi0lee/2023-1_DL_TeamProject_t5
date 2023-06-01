import math
import random
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning.callbacks import Callback
from PIL import Image
from torchvision.transforms import ToPILImage
import wandb


class InverseSqrtScheduler(LambdaLR):

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            
            decay_factor = warmup_steps ** 0.5
            return decay_factor * step ** -0.5

        super(InverseSqrtScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
 
class LogPredictionsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.random_indices = None
        self.to_pil = ToPILImage()

    def on_validation_epoch_start(self, trainer, pl_module):

        self.random_indices = []
        num_indices = 10
        # math.ceil(len(trainer.val_dataloaders) * 0.005)

        self.random_indices = list(random.sample(range(len(trainer.val_dataloaders)), num_indices))
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch: Any, batch_idx: int, dataloader_idx: int = 0):

        if batch_idx in self.random_indices:
            n = 1
            pixel_values, _, target_sequences = batch
            images = [self.to_pil(img) for img in pixel_values[0]]

            # captions = [f'Ground Truth: {y_i}\nPrediction: {y_pred}' for y_i, y_pred in zip(target_sequences[:n], outputs[:n])]
            # trainer.logger.log_image(key='sample_images', images=images, caption=captions)

            columns = ['image', 'ground truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(images, target_sequences[:n], outputs[:n]))]
            trainer.logger.log_table(key='sample_table', columns=columns, data=data)
