import os, pprint
import torch
import loralib
from KD_Lib.models import model_dict
from KD_Lib.models.resnet_torch import get_ResNet, monkey_patch
from KD_Lib.models.shake import ShakeHead
from KD_Lib.KD import VanillaKD, DML, Shake, Smooth, FNKD, TriKD
from KD_Lib.datasets import get_dataset, get_cifar100_dataloaders
from KD_Lib.KD.common.loss import KLDivLoss, SharpLoss
from KD_Lib.utils import get_optim_sched, log_cfg

def train(cfg, logger=None, trial=None):
    logger.save_log(pprint.pformat(cfg.__dict__))

    # Dataset
    if cfg.DATASET == 'cifar100':
        loaders = get_cifar100_dataloaders(cfg.DATA_PATH, batch_size=cfg.BATCH_SIZE, num_workers=cfg.WORKERS)
        cfg.CLASSES = 100
    else:
        loaders = get_dataset(cfg)

    # Models
    if cfg.CUSTOM_MODEL:
        models = [model_dict[cfg.TEACHER](num_classes=cfg.CLASSES),
                  model_dict[cfg.STUDENT](num_classes=cfg.CLASSES)]
        models = [monkey_patch(model, custom=cfg.CUSTOM_MODEL).to(cfg.DEVICE) for model in models]
    else:
        models = [get_ResNet(model=cfg.TEACHER, cfg=cfg).to(cfg.DEVICE),
                  get_ResNet(model=cfg.STUDENT, cfg=cfg).to(cfg.DEVICE)]

    # Optimizers and schedulers
    optim_sched = get_optim_sched(models, cfg)
    optimizers, schedulers = optim_sched['optims'], optim_sched['scheds']

    # Losses()
    losses = [torch.nn.CrossEntropyLoss(reduction='mean').to(cfg.DEVICE),
              KLDivLoss(cfg, reduction='batchmean').to(cfg.DEVICE)]

    # Training
    if cfg.MODE == 'kd': # Vanilla KD
        distiller = VanillaKD(models, loaders, optimizers, schedulers, losses, cfg)
        distiller.train_student()
        distiller.evaluate(teacher=False, verbose=True) # Evaluate the student network

    elif cfg.MODE == 'fnkd': # Vanilla KD
        distiller = FNKD(models, loaders, optimizers, schedulers, losses, cfg)
        distiller.train_student()
        distiller.evaluate(teacher=False, verbose=True) # Evaluate the student network

    elif cfg.MODE == 'dml': # DML
        distiller = DML(models, loaders, optimizers, schedulers, losses, cfg)
        distiller.train_student()
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'ftkd': # Fine-tuning
        distiller = DML(models, loaders, optimizers, schedulers, losses, cfg)
        distiller.train_student(stud_teach_kd=False)
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'shake': # SHAKE
        data = torch.randn(2, 3, cfg.IMSIZE, cfg.IMSIZE).to(cfg.DEVICE)
        _, feat_t, _, _ = models[0](data, return_feats=True)
        shake = ShakeHead(feat_t).to(cfg.DEVICE)
        models.insert(1, shake)

        optim_sched = get_optim_sched(models[1:], cfg, single=True)
        optimizers[1], schedulers[1] = optim_sched['optims'], optim_sched['scheds']

        distiller = Shake(models, loaders, optimizers, schedulers, losses, cfg)
        
        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            distiller.train_teacher()
        else:
            distiller.models[0].load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))
        t_val, _ = distiller.evaluate(teacher=True)
        logger.save_log(f"Teacher Accuracy: {t_val:.4f}%")

        distiller.train_student()
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'smooth': # New method
        if cfg.LORA:
            smooth = loralib.Linear(64, cfg.CLASSES, r=16).to(cfg.DEVICE)
        else:
            smooth = torch.nn.Linear(64, cfg.CLASSES).to(cfg.DEVICE)
        models.insert(1, smooth)

        optim_sched = get_optim_sched(models[1:], cfg, single=True)
        optimizers[1], schedulers[1] = optim_sched['optims'], optim_sched['scheds']
        losses.append(SharpLoss())

        distiller = Smooth(models, loaders, optimizers, schedulers, losses, cfg, logger, trial)

        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            distiller.train_teacher()
        else:
            distiller.models[0].load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))
        if cfg.PRETRAINED_HEAD and not cfg.LORA:
            distiller.models[1].weight.data = list(distiller.models[0].modules())[-1].weight.data.clone()
            distiller.models[1].bias.data = list(distiller.models[0].modules())[-1].bias.data.clone()

        t_val, _ = distiller.evaluate(teacher=True)
        logger.save_log(f"Teacher Accuracy: {t_val:.4f}%")

        distiller.train_student()
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'trikd': # TriKD
        smooth = torch.nn.Linear(64, cfg.CLASSES).to(cfg.DEVICE)
        models.insert(1, smooth)

        anchor = model_dict[cfg.STUDENT](num_classes=cfg.CLASSES)
        anchor = monkey_patch(anchor, custom=cfg.CUSTOM_MODEL).to(cfg.DEVICE)
        anchor.load_state_dict(torch.load(cfg.STUDENT_WEIGHTS))
        models.insert(0, anchor)

        optim_sched = get_optim_sched(models[2:], cfg, single=True)
        optimizers[1], schedulers[1] = optim_sched['optims'], optim_sched['scheds']

        distiller = TriKD(models, loaders, optimizers, schedulers, losses, cfg)

        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            distiller.train_teacher()
        else:
            distiller.models[1].load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))
        
        t_val, _ = distiller.evaluate(teacher=True)
        logger.save_log(f"Teacher Accuracy: {t_val:.4f}%")
        a_val, _ = distiller.evaluate(anchor=True)
        logger.save_log(f"Anchor Accuracy: {a_val:.4f}%")

        distiller.train_student()
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'baseline': # Baseline
        distiller = VanillaKD(models, loaders, optimizers, schedulers, losses, cfg)
        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            distiller.train_teacher()
        else:
            distiller.teacher_model.load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))        
        distiller.evaluate(teacher=True, verbose=True) # Evaluate the student network

    else: # Error
        raise NotImplementedError(f"{cfg.MODE} is not implemented!")
    
    # Save config
    log_cfg(distiller.cfg)
    return cfg.VACC['S_BEST']
