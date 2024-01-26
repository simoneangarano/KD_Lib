import torch.nn.functional as F
from KD_Lib.KD.common import BaseClass

class VanillaKD(BaseClass):
    """ Original implementation of Knowledge distillation from the paper "Distilling the
    Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf """
    def __init__(self, models, loaders, optimizers, schedulers, losses, cfg):
        super(VanillaKD, self).__init__(models, loaders, optimizers, schedulers, losses, cfg)

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """ Function used for calculating the KD loss during distillation """
        ce_loss = self.loss_ce(y_pred_student, y_true)
        kd_loss = self.loss_kd(y_pred_student, y_pred_teacher)
        loss = ce_loss + self.cfg.W*kd_loss
        return loss, ce_loss, kd_loss