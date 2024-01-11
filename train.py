import os, json
from pprint import pformat
import torch
from KD_Lib.models import model_dict
from KD_Lib.models.resnet_torch import get_ResNet, monkey_patch
from KD_Lib.models.shake import ShakeHead
from KD_Lib.KD import VanillaKD, DML, Shake, Smooth
from KD_Lib.datasets import get_dataset, get_cifar100_dataloaders
from KD_Lib.utils import get_optim_sched
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'

# Hyperparameters
class Cfg:
    def __init__(self, dict=None):
        if dict is not None:
            for key in dict:
                setattr(self, key, dict[key])
            return
        
        self.MODE: str = 'smooth' # 'kd' or 'dml' or 'shake' or 'smooth' or 'baseline'
        self.DATASET: str = 'cifar100' # 'cifar10' or 'cifar100'
        self.IMSIZE: int = 32 if 'cifar' in self.DATASET else 227
        self.CLASSES: int = 0
        self.DATA_PATH: str = '../Knowledge-Distillation-Zoo/datasets/'
        self.BATCH_SIZE: int = 64
        self.WORKERS: int = 8
        self.TEACHER: str = 'resnet110' 
        self.STUDENT: str = 'resnet20'
        self.CUSTOM_MODEL: bool = True
        self.LAYER_NORM: bool = False
        self.LR: float = 0.05
        self.LR_MIN: float = 5e-5
        self.MOMENTUM: float = 0.9
        self.WD: float = 5e-4
        self.T: float = 4.0
        self.W: float = 1.0
        self.FEAT_NORM: bool = False
        self.EPOCHS: int = 240
        self.SCHEDULER: str = 'step' # 'cos' or 'step' or 'lin'
        self.STEPS: list = [150, 180, 210]
        self.GAMMA: float = 0.1
        self.TEACHER_WEIGHTS: str = f'./models/teacher_{self.DATASET}_kd.pt'
        self.PARALLEL: bool = False
        self.EXP: str = f"{self.MODE}_{self.DATASET}"
        self.LOG: bool = True
        self.LOG_DIR: str = f"./tb/{self.EXP}/"
        self.SAVE_PATH: str = f"./models/{self.EXP}.pt"
        self.DEVICE: str = 'cuda'
        self.VACC: dict = {
            'T_LAST': 0.0,
            'T_BEST': 0.0,
            'S_LAST': 0.0,
            'S_BEST': 0.0
        }

def main():
    cfg = Cfg()
    print(pformat(cfg.__dict__))

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
    losses = [torch.nn.CrossEntropyLoss(reduction='mean').to(cfg.DEVICE), ######################################
              torch.nn.KLDivLoss(reduction='batchmean', log_target=True).to(cfg.DEVICE)] ####################### 

    # Training
    if cfg.MODE == 'kd': # Vanilla KD
        distiller = VanillaKD(models, loaders, optimizers, schedulers, losses, cfg)
        distiller.train_student()
        distiller.evaluate(teacher=False, verbose=True) # Evaluate the student network

    elif cfg.MODE == 'dml': # DML
        distiller = DML(models, loaders, optimizers, schedulers, losses, cfg)
        distiller.train_students()
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'shake': # SHAKE
        data = torch.randn(2, 3, cfg.IMSIZE, cfg.IMSIZE).cuda()
        _, feat_t, _, _ = models[0](data, return_feats=True)
        shake = ShakeHead(feat_t).to('cuda')
        models.insert(1, shake)

        optim_sched = get_optim_sched(models[1:], cfg, single=True)
        optimizers[1], schedulers[1] = optim_sched['optims'], optim_sched['scheds']

        distiller = Shake(models, loaders, optimizers, schedulers, losses, cfg)
        
        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            distiller.train_teacher()
        else:
            distiller.models[0].load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))
        cfg.VACC['T_BEST'], _ = distiller.evaluate(teacher=True)
        print(f"Teacher Accuracy: {cfg.VACC['T_BEST']:.4f}%")

        distiller.train_students()
        distiller.evaluate(verbose=True)

    elif cfg.MODE == 'smooth': # New method
        smooth = torch.nn.Linear(2048, cfg.CLASSES).to(cfg.DEVICE)
        models.insert(1, smooth)

        optim_sched = get_optim_sched(models[1:], cfg, single=True)
        optimizers[1], schedulers[1] = optim_sched['optims'], optim_sched['scheds']

        distiller = Smooth(models, loaders, optimizers, schedulers, losses, cfg)

        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            distiller.train_teacher()
        else:
            distiller.models[0].load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))
        distiller.models[1].weight.data = list(distiller.models[0].modules())[-1].weight.data.clone()
        distiller.models[1].bias.data = list(distiller.models[0].modules())[-1].bias.data.clone()

        cfg.VACC['T_BEST'], _ = distiller.evaluate(teacher=True)
        print(f"Teacher Accuracy: {cfg.VACC['T_BEST']:.4f}%")

        distiller.train_students()
        distiller.evaluate(verbose=True)

    else: # Baseline
        distiller = VanillaKD(models, loaders, optimizers, schedulers, losses, cfg)
        if not os.path.exists(cfg.TEACHER_WEIGHTS):
            distiller.train_teacher()
        else:
            distiller.teacher_model.load_state_dict(torch.load(cfg.TEACHER_WEIGHTS))        
        distiller.evaluate(teacher=True, verbose=True) # Evaluate the student network

    # Save config
    if not os.path.exists(f"./exp/"):
        os.makedirs(f"./exp/")
    with open(f"./exp/{cfg.EXP}.json", "w") as file:
        json.dump(distiller.cfg.__dict__, file)

if __name__ == "__main__":
    main()