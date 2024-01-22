from datetime import datetime

from KD_Lib.hp_search import HPSearcher
from KD_Lib.train import train
from KD_Lib.utils import Logger

# Hyperparameters
class Cfg:
    def __init__(self, dict=None):
        if dict is not None:
            for key in dict:
                setattr(self, key, dict[key])
            return
        
        self.MODE: str = 'smooth' # 'kd' or 'dml' or 'shake' or 'smooth' or 'fnkd' or 'baseline' or 'ftkd' or 'trikd'
        self.DATASET: str = 'cifar100' # 'cifar10' or 'cifar100'
        self.EXP: str = f"{self.MODE}_{self.DATASET}_hp"

        self.T: float = 1.0 if self.MODE in ['trikd'] else 4.0
        self.W: float = 0.9 if self.MODE == 'kd' else 9.0 if self.MODE in ['fnkd'] else 1.0
        self.L = [1,1,1,1,1,1] # La, Lb, Lc, Ld, Le, Lf
        self.FEAT_NORM: bool = True if self.MODE in ['fnkd'] else False

        self.IMSIZE: int = 32 if 'cifar' in self.DATASET else 227
        self.CLASSES: int = 0
        self.DATA_PATH: str = '../Knowledge-Distillation-Zoo/datasets/'
        self.BATCH_SIZE: int = 64
        self.WORKERS: int = 8
        self.TEACHER: str = 'resnet110' 
        self.STUDENT: str = 'resnet20'
        self.CUSTOM_MODEL: bool = True
        self.LAYER_NORM: bool = False
        self.LR: float = 0.1 if self.MODE in ['trikd'] else 0.05
        self.LR_MIN: float = 5e-5
        self.MOMENTUM: float = 0.9
        self.WD: float = 5e-4
        self.EPOCHS: int = 1
        self.SCHEDULER: str = 'step' # 'cos' or 'step' or 'lin'
        self.STEPS: list = [150, 180, 210]
        self.GAMMA: float = 0.1
        self.TEACHER_WEIGHTS: str = f'./models/{self.TEACHER}_{self.DATASET}.pt'
        self.STUDENT_WEIGHTS: str = f'./models/{self.STUDENT}_{self.DATASET}.pt'
        self.PRETRAINED_HEAD: bool = False
        self.PARALLEL: bool = False
        self.LOG: bool = True
        self.LOG_DIR: str = f"./exp/"
        self.TB_DIR: str = f"./tb/{self.EXP}/"
        self.SAVE_PATH: str = f"./models/{self.EXP}.pt"
        self.DEVICE: str = 'cuda'
        self.TIME: float = 0.0
        self.HP_SEARCH_DIR: str = './hp_search/'
        self.TRIALS: int = 2
        self.VACC: dict = {
            'T_LAST': 0.0,
            'T_BEST': 0.0,
            'S_LAST': 0.0,
            'S_BEST': 0.0
        }

def main():
    hp_search = True # make it argparser
    cfg = Cfg()
    if hp_search:
        name = f'hp_search_{datetime.now().strftime("%y%m%d%H%M")}'
        logger = Logger(f"{cfg.HP_SEARCH_DIR}{name}.txt")
        searcher = HPSearcher(cfg, name, logger)
        searcher.hp_search()
    else:
        train(cfg)

if __name__ == "__main__":
    main()