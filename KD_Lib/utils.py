import os, json
import numpy as np
import torch

def get_optim_sched(models: list, cfg, single=False):
    os = {'optims': [], 'scheds': []}
        
    if single:
        params = torch.nn.ModuleList(models).parameters()
        opt, sched = get_single_opt_sched(params, cfg)
        os['optims'] = opt
        os['scheds'] = sched
        return os
    
    for model in models:
        params = list(model.parameters())
        opt, sched = get_single_opt_sched(model.parameters(), cfg)
        os['optims'].append(opt)
        os['scheds'].append(sched)
    return os

def get_single_opt_sched(params, cfg):
    opt = torch.optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WD)
    if cfg.SCHEDULER == 'cos':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
    elif cfg.SCHEDULER == 'step':
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.STEPS, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'lin':
        sched = torch.optim.lr_scheduler.LinearLR(opt, total_iters=cfg.EPOCHS, start_factor=1, end_factor=cfg.LR_MIN/cfg.LR)
    return opt, sched

def sharpness(logits, eps=1e-9, clip=70):
    """Computes the sharpness of the logits.
    Args:
        logits: Tensor of shape [batch_size, num_classes] containing the logits.
        eps: Small epsilon to avoid numerical issues.
    Returns:
        The sharpness of the logits.
    """
    logits = logits.detach().cpu().numpy()
    if clip != np.inf:
        logits = np.clip(logits, -clip, clip)
    else: 
        logits = logits.astype(np.float128)
    return np.mean(np.log(np.exp(logits).sum(axis=1) + eps))

def sharpness_gap(teacher_logits, student_logits, eps=1e-9):
    """Computes the sharpness gap between the teacher and student logits.
    Args:
        teacher_logits: Tensor of shape [batch_size, num_classes] containing the teacher logits.
        student_logits: Tensor of shape [batch_size, num_classes] containing the student logits.
        eps: Small epsilon to avoid numerical issues.
    Returns:
        The sharpness gap between the teacher and student logits.
    """
    teacher_sharpness = sharpness(teacher_logits, eps)
    student_sharpness = sharpness(student_logits, eps)
    return teacher_sharpness - student_sharpness, teacher_sharpness, student_sharpness

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def log_cfg(cfg):
    if not os.path.exists(cfg.LOG_DIR):
        os.makedirs(cfg.LOG_DIR)
    with open(f"{cfg.LOG_DIR}{cfg.EXP}.json", "w") as file:
        json.dump(cfg.__dict__, file)

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

class SharpLoss(torch.nn.MSELoss):
    def __init__(self, max_sharpness=None, reduction='mean'):
        super(SharpLoss, self).__init__(reduction=reduction)

    def forward(self, teacher_logits, student_logits):
        teacher_sharpness = sharpness_torch(teacher_logits)
        student_sharpness = sharpness_torch(student_logits)
        return super(SharpLoss, self).forward(student_sharpness, teacher_sharpness)