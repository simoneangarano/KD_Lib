import numpy as np
import torch
import torch.nn.functional as F

class KLDivLoss(torch.nn.Module):
    def __init__(self, cfg, reduction='batchmean'):
        super(KLDivLoss, self).__init__()
        self.log_target = cfg.LOG_TARGET
        self.T = cfg.T
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.kld_loss = torch.nn.KLDivLoss(reduction=reduction, log_target=self.log_target)

    def forward(self, outputs, targets): # logit, logit
        outputs = self.log_softmax(outputs/self.T)
        if self.log_target:
            targets = self.log_softmax(targets.detach()/self.T)
        else:
            targets = self.softmax(targets.detach()/self.T)
        loss = self.T * self.T * self.kld_loss(outputs, targets)
        return loss
    
class SharpLoss(torch.nn.MSELoss):
    def __init__(self, max_sharpness=None, reduction='mean'):
        super(SharpLoss, self).__init__(reduction=reduction)

    def forward(self, teacher_logits, student_logits):
        teacher_sharpness = sharpness_torch(teacher_logits)
        student_sharpness = sharpness_torch(student_logits)
        return super(SharpLoss, self).forward(student_sharpness, teacher_sharpness)
    
class JocorLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(JocorLoss, self).__init__()
        self.co_lambda = cfg.CO_LAMBDA
        self.T = cfg.T
        self.forget_scheduler = np.ones(cfg.EPOCHS) * cfg.FORGET_RATE
        self.forget_scheduler[:cfg.GRADUAL] = np.linspace(0, cfg.FORGET_RATE**cfg.EXPONENT, cfg.GRADUAL)

    def forward(self, y_s, y_t, y, epoch):
        ce_s = F.cross_entropy(y_s, y, reduction='none')
        ce_t = F.cross_entropy(y_t, y, reduction='none')
        kl_s = self.T * self.T * kl_loss_compute(y_s/self.T, y_t.detach()/self.T, reduction='none')
        kl_t = self.T * self.T * kl_loss_compute(y_t/self.T, y_s.detach()/self.T, reduction='none')
        loss = ((1-self.co_lambda) * (ce_s+ce_t) + self.co_lambda * (kl_s+kl_t)).cpu()

        ind_sorted = np.argsort(loss.data)
        loss_sorted = loss[ind_sorted]
        remember_rate = 1 - self.forget_scheduler[epoch]
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
        loss = torch.mean(loss[ind_update])
        return loss

class SmoothLoss(torch.nn.Module):
    def __init__(self, cfg):
        super(SmoothLoss, self).__init__()
        self.JOCOR = cfg.JOCOR
        self.W = cfg.W
        self.T = cfg.T
        self.L = cfg.L
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='none').to(cfg.DEVICE)
        self.loss_kd = KLDivLoss(cfg, reduction='none').to(cfg.DEVICE)
        if self.JOCOR:
            self.forget_scheduler = np.zeros(cfg.EPOCHS)
            self.forget_scheduler[:cfg.GRADUAL] = np.linspace(cfg.FORGET_RATE, 0, cfg.GRADUAL)

    def forward(self, y_s, y_t, y_ts, y, ep):
        # classification loss student
        loss_cls = self.loss_ce(y_s, y)
        # distillation loss student <-> smooth head
        La = self.loss_kd(y_ts, y_s).sum(dim=1)
        Lb = self.loss_kd(y_s, y_ts).sum(dim=1)
        # classification loss smooth head
        Lc = self.loss_ce(y_ts, y)
        # distillation loss smooth head <-> teacher
        Ld = F.mse_loss(y_ts, y_t.detach(), reduction='none').mean(dim=1)
        # distillation loss student <-> teacher
        Le = self.loss_kd(y_s, y_t).sum(dim=1)
        # sharpness loss
        # Lf = self.loss_sharp(y_ts, y_s.detach())

        loss_kd = self.L[0]*La + self.L[1]*Lb + self.L[2]*Lc + self.L[3]*Ld + self.L[4]*Le # + self.L[5]*Lf
        loss = loss_cls + self.W*loss_kd

        if self.JOCOR and self.forget_scheduler[ep] > 0:
            loss = loss.cpu()
            ind_sorted = np.argsort(loss.data)
            loss_sorted = loss[ind_sorted]
            remember_rate = 1 - self.forget_scheduler[ep]
            num_remember = int(remember_rate * len(loss_sorted))
            ind_update = ind_sorted[:num_remember]
            loss = loss[ind_update]
            loss_cls = loss_cls[ind_update]
            loss_kd = loss_kd[ind_update]
        return torch.mean(loss), torch.mean(loss_cls), torch.mean(loss_kd)

class MWKDLoss(SmoothLoss):
    def __init__(self, cfg):
        super(MWKDLoss, self).__init__(cfg)

    def forward(self, y_s, y_t, y_ss, y, ep):
        # classification loss student
        loss_cls = self.loss_ce(y_s, y)
        # distillation loss student <-> smooth head
        La = self.loss_kd(y_ss, y_s).sum(dim=1)
        Lb = self.loss_kd(y_s, y_ss).sum(dim=1)
        # distillation loss student <-> teacher
        Le = self.loss_kd(y_ss, y_t.detach()).sum(dim=1)

        loss_kd = self.L[0] * La + self.L[1] * Lb + self.L[4] * Le
        loss = loss_cls + self.W * loss_kd

        return torch.mean(loss), torch.mean(loss_cls), torch.mean(loss_kd)

def sharpness_torch(logits):
    """Computes the sharpness of the logits.
    Args:
        logits: Tensor of shape [batch_size, num_classes] containing the logits.
        eps: Small epsilon to avoid numerical issues.
    Returns:
        The sharpness of the logits.
    """
    logits = torch.exp(logits).sum(dim=1).log()
    return logits

def kl_loss_compute(pred, soft_targets, reduction='none'):
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1),reduction='none')
    if reduction:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)