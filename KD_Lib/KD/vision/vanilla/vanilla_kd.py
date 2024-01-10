import torch.nn.functional as F
from KD_Lib.KD.common import BaseClass

class VanillaKD(BaseClass):
    """
    Original implementation of Knowledge distillation from the paper "Distilling the
    Knowledge in a Neural Network" https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(
        self,
        models,
        loaders,
        optimizers,
        schedulers,
        losses,
        cfg
    ):
        super(VanillaKD, self).__init__(
            models,
            loaders,
            optimizers,
            schedulers,
            losses,
            cfg
        )

    def calculate_kd_loss(self, y_pred_student, y_pred_teacher, y_true):
        """
        Function used for calculating the KD loss during distillation
        """

        soft_teacher_out = F.log_softmax(y_pred_teacher / self.cfg.T, dim=1)
        log_soft_student_out = F.log_softmax(y_pred_student / self.cfg.T, dim=1)

        ce_loss = self.loss_ce(y_pred_student, y_true)
        kd_loss = self.cfg.W * self.cfg.T * self.cfg.T * self.loss_kd(log_soft_student_out, soft_teacher_out)
        loss = ce_loss + kd_loss
        return loss, ce_loss, kd_loss
