from datetime import datetime
import argparse, os

import random, numpy as np
from KD_Lib.hp_search import HPTuner
from KD_Lib.train import train
from KD_Lib.utils import Logger, set_environment
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4,5,6,7'
os.environ['RAY_DEDUP_LOGS'] = '0'

# Hyperparameters
class Cfg:
    def __init__(self, dict=None, name=None, seed=-1):
        if dict is not None:
            for key in dict:
                setattr(self, key, dict[key])
            return
        
        # Important
        self.MODE: str = 'mwkd' # 'kd' or 'dml' or 'shake' or 'smooth' or 'fnkd' or 'baseline' or 'ftkd' or 'trikd'
        self.DATASET: str = 'cifar100' # 'cifar10' or 'cifar100'
        self.T: float = 1.0 if self.MODE in ['trikd'] else 4.0
        self.W: float = 0.9 if self.MODE == 'kd' else 9.0 if self.MODE in ['fnkd'] else 1.0
        self.L = [0, 0, 0, 0, 1, 0] # La, Lb, Lc, Ld, Le, Lf
        # Jocor Loss
        self.JOCOR: bool = False
        self.GRADUAL: int = 180
        self.FORGET_RATE: float = 0.1
        # Dataset
        self.IMSIZE: int = 32 if 'cifar' in self.DATASET else 227
        self.CLASSES: int = 0
        self.DATA_PATH: str = os.path.abspath('../Knowledge-Distillation-Zoo/datasets/')
        self.BATCH_SIZE: int = 64
        self.WORKERS: int = 8
        # Models
        self.TEACHER: str = 'resnet110' 
        self.STUDENT: str = 'resnet20'
        self.PRETRAINED: bool = False
        self.PRETRAINED_HEAD: bool = False
        self.LORA: bool = False
        self.CUSTOM_MODEL: bool = True
        self.LAYER_NORM: bool = False
        self.FEAT_NORM: bool = True if self.MODE in ['fnkd'] else False
        self.LOG_TARGET: bool = False
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
        self.LOG: bool = True
        self.CWD: str = os.getcwd()
        self.NAME: str = f"_{name}" if name is not None else ""
        self.EXP: str = f"{self.MODE}_{self.TEACHER}_{self.STUDENT}_{self.DATASET}{self.NAME}"
        self.TEACHER_WEIGHTS: str = os.path.abspath(f'./models/{self.TEACHER}_{self.DATASET}.pt')
        self.STUDENT_WEIGHTS: str = os.path.abspath(f'./models/{self.STUDENT}_{self.DATASET}.pt')
        self.LOG_DIR: str = os.path.abspath(f"./exp/")
        self.TB_DIR: str = os.path.abspath(f"./tb/{self.EXP}/")
        self.SAVE_PATH: str = os.path.abspath(f"./models/{self.EXP}.pt")
        self.HP_SEARCH_DIR: str = os.path.abspath('./hp_search/')
        # Runtime
        self.PARALLEL: bool = False
        self.DEVICE: str = 'cuda'
        self.SEED: int = seed if seed != 0 else random.randint(0,1000000)
        # Metrics
        self.TRIAL: int = 0
        self.TIME: float = 0.0
        self.VACC: dict = {'T_LAST': 0.0, 'T_BEST': 0.0, 'S_LAST': 0.0, 'S_BEST': 0.0}
        # HP Search
        self.N_TRIALS: int = 27
        self.SEARCH_SPACE: dict = {
            "La": [1],                  # KLDiv Ts->S [1]
            "Lb": [1],                  # KLDiv S->Ts [1]
            "Lc": [0.1, 0.3, 1],        # CE Ts       [1, 2]
            "Ld": [0.01, 0.03, 0.1],    # MSE Ts->T   [0.1, 0.5]
            "Le": [0.01, 0.03, 0.1],    # KLDiv S->T  [0.5, 0.3]
            "Lf": [0],                  # SharpLoss   [0, 0.01, 0.1]
            "T":  [4],                  # Temperature [4]
            "W":  [1],                  # KD          [1]
        }

    def reset(self):
        self.TB_DIR: str = os.path.join(self.CWD, f"./tb/{self.EXP}/")
        self.SAVE_PATH: str = os.path.join(self.CWD, f"models/{self.EXP}.pt")
        self.TIME: float = 0.0
        self.VACC: dict = {'T_LAST': 0.0, 'T_BEST': 0.0, 'S_LAST': 0.0, 'S_BEST': 0.0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hp_search', '-s', action='store_true')
    parser.add_argument('--device', '-d', type=str, default='-1')
    parser.add_argument('--rand_seed', '-r', type=int, default=-1)
    parser.add_argument('--cfg', '-c', type=str, default=None)
    parser.add_argument('--name', '-n', type=str, default=None)
    parser.add_argument('--trials', '-t', type=int, default=1)
    args = parser.parse_args()
    
    if args.hp_search:
        cfg = Cfg(args.cfg, args.name, args.rand_seed)
        set_environment(cfg.SEED, args.device)
        name = f'hp_search_{datetime.now().strftime("%y%m%d%H%M%S")}'
        logger = Logger(os.path.join(cfg.HP_SEARCH_DIR, f"{name}.txt"))
        searcher = HPTuner(cfg, name, logger)
        searcher.hp_search()
        return
        
    val_accs = []
    for t in range(args.trials):
        t_name = args.name + f'_{t}' if args.name is not None else str(t)
        cfg = Cfg(args.cfg, t_name, args.rand_seed)
        set_environment(cfg.SEED, args.device)

        name = f'{cfg.EXP}_{datetime.now().strftime("%y%m%d%H%M%S")}'
        logger = Logger(os.path.join(cfg.LOG_DIR, f"{name}.txt"))
        logger.save_log(f"Trial {t+1}/{args.trials}")
        val_accs.append(train(cfg, logger))
    val_accs = np.array(val_accs)
    logger.save_log(f"Best Val Acc on {args.trials} trials: Avg {val_accs.mean():.4f}, Std: {val_accs.std():.4f}")

if __name__ == "__main__":
    main()