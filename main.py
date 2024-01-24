from datetime import datetime
import os, argparse

import torch, random, numpy as np
from KD_Lib.hp_search import HPSearcher
from KD_Lib.train import train
from KD_Lib.utils import Logger

# Hyperparameters
class Cfg:
    def __init__(self, dict=None, name=None):
        if dict is not None:
            for key in dict:
                setattr(self, key, dict[key])
            return
        
        # Important
        self.MODE: str = 'smooth' # 'kd' or 'dml' or 'shake' or 'smooth' or 'fnkd' or 'baseline' or 'ftkd' or 'trikd'
        self.DATASET: str = 'cifar100' # 'cifar10' or 'cifar100'
        self.NAME: str = f"_{name}" if name is not None else ""

        self.T: float = 1.0 if self.MODE in ['trikd'] else 4.0
        self.W: float = 0.9 if self.MODE == 'kd' else 9.0 if self.MODE in ['fnkd'] else 1.0
        self.L = [1,1,2,0.1,0.3,0.1] # La, Lb, Lc, Ld, Le, Lf
        self.FEAT_NORM: bool = True if self.MODE in ['fnkd'] else False
        self.JOCOR: bool = False
        self.EXPONENT: float = 1.0
        self.GRADUAL: int = 10
        self.FORGET_RATE: float = 0.25
        self.CO_LAMBDA: float = 0.1
        # Dataset
        self.IMSIZE: int = 32 if 'cifar' in self.DATASET else 227
        self.CLASSES: int = 0
        self.DATA_PATH: str = '../Knowledge-Distillation-Zoo/datasets/'
        self.BATCH_SIZE: int = 64
        self.WORKERS: int = 8
        # Models
        self.TEACHER: str = 'resnet110' 
        self.STUDENT: str = 'resnet20'
        self.PRETRAINED_HEAD: bool = False
        self.CUSTOM_MODEL: bool = True
        self.LAYER_NORM: bool = False
        # Training
        self.LR: float = 0.1 if self.MODE in ['trikd'] else 0.05
        self.LR_MIN: float = 5e-5
        self.MOMENTUM: float = 0.9
        self.WD: float = 5e-4
        self.EPOCHS: int = 240
        self.SCHEDULER: str = 'step' # 'cos' or 'step' or 'lin'
        self.STEPS: list = [150, 180, 210]
        self.GAMMA: float = 0.1
        # Paths and Logging
        self.EXP: str = f"{self.MODE}_{self.TEACHER}_{self.STUDENT}_{self.DATASET}{self.NAME}"
        self.TEACHER_WEIGHTS: str = f'./models/{self.TEACHER}_{self.DATASET}.pt'
        self.STUDENT_WEIGHTS: str = f'./models/{self.STUDENT}_{self.DATASET}.pt'
        self.LOG: bool = True
        self.LOG_DIR: str = f"./exp/"
        self.TB_DIR: str = f"./tb/{self.EXP}/"
        self.SAVE_PATH: str = f"./models/{self.EXP}.pt"
        self.HP_SEARCH_DIR: str = './hp_search/'
        # Runtime
        self.PARALLEL: bool = False
        self.DEVICE: str = 'cuda'
        self.SEED: int = 42
        # Metrics
        self.TIME: float = 0.0
        self.VACC: dict = {'T_LAST': 0.0, 'T_BEST': 0.0, 'S_LAST': 0.0, 'S_BEST': 0.0}
        # HP Search
        self.TRIALS: int = 12
        self.SEARCH_SPACE: dict = {
            "La": [1],              # KLDiv Ts->S [1]
            "Lb": [1],              # KLDiv S->Ts [1]
            "Lc": [1, 2, 3],        # CE Ts       [1, 2]
            "Ld": [0.01, 0.1],       # MSE Ts->T  [0.1, 0.5]
            "Le": [0.5, 0.3],       # KLDiv S->T  [0.5, 0.3]
            "Lf": [0],              # SharpLoss   [0, 0.01, 0.1]
            "T": [4],               # Temperature [4]
            "W": [1],               # KD          [1]
        }

    def reset(self):
        self.TB_DIR: str = f"./tb/{self.EXP}/"
        self.SAVE_PATH: str = f"./models/{self.EXP}.pt"
        self.TIME: float = 0.0
        self.VACC: dict = {'T_LAST': 0.0, 'T_BEST': 0.0, 'S_LAST': 0.0, 'S_BEST': 0.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_search', '-s', action='store_true')
    parser.add_argument('--device', '-d', type=str, default='-1')
    parser.add_argument('--cfg', '-c', type=str, default=None)
    parser.add_argument('--name', '-n', type=str, default=None)
    args = parser.parse_args()
    cfg = Cfg(args.cfg, args.name)

    if int(args.device) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    if cfg.SEED >= 0:
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)
        random.seed(cfg.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.hp_search:
        name = f'hp_search_{datetime.now().strftime("%y%m%d%H%M%S")}'
        logger = Logger(f"{cfg.HP_SEARCH_DIR}{name}.txt")
        searcher = HPSearcher(cfg, name, logger)
        searcher.hp_search()
    else:
        name = f'{cfg.EXP}_{datetime.now().strftime("%y%m%d%H%M%S")}'
        logger = Logger(f"{cfg.LOG_DIR}{name}.txt")
        train(cfg, logger)

if __name__ == "__main__":
    main()