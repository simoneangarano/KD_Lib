import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, LinearLR

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
    opt = optim.SGD(params, lr=cfg.LR, momentum=cfg.MOMENTUM, weight_decay=cfg.WD)
    if cfg.SCHEDULER == 'cos':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS, eta_min=cfg.LR_MIN)
    elif cfg.SCHEDULER == 'step':
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.STEPS, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'lin':
        sched = torch.optim.lr_scheduler.LinearLR(opt, total_iters=cfg.EPOCHS, start_factor=1, end_factor=cfg.LR_MIN/cfg.LR)
    return opt, sched