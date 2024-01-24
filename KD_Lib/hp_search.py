import os
import joblib
import optuna

from KD_Lib.train import train
from KD_Lib.utils import MultiPruner


class HPSearcher:
      
    def __init__(self, cfg, name, logger=None, trial=None):
        
        self.cfg = cfg
        self.name = name
        self.logger = logger
        self.trial = trial
            
    def get_random_hps(self):
        self.cfg.T = self.trial.suggest_categorical("T", self.cfg.SEARCH_SPACE["T"])
        self.cfg.W = self.trial.suggest_categorical("W", self.cfg.SEARCH_SPACE["W"])
        self.cfg.L = [self.trial.suggest_categorical("La", self.cfg.SEARCH_SPACE["La"]),
                      self.trial.suggest_categorical("Lb", self.cfg.SEARCH_SPACE["Lb"]),
                      self.trial.suggest_categorical("Lc", self.cfg.SEARCH_SPACE["Lc"]),
                      self.trial.suggest_categorical("Ld", self.cfg.SEARCH_SPACE["Ld"]),
                      self.trial.suggest_categorical("Le", self.cfg.SEARCH_SPACE["Le"]),
                      self.trial.suggest_categorical("Lf", self.cfg.SEARCH_SPACE["Lf"])]
        print(f"T={self.cfg.T}, W={self.cfg.W}, L={self.cfg.L}")
    
    def objective(self, trial):
        self.cfg.EXP = f"{self.name}_{trial.number}"
        self.cfg.reset()
        self.trial = trial 
        self.get_random_hps()

        metr = train(cfg=self.cfg, logger=self.logger, trial=self.trial)
        return metr
    
    
    def hp_search(self):
        self.study = optuna.create_study(study_name=self.name,
                                         direction='maximize', 
                                         sampler=optuna.samplers.GridSampler(self.cfg.SEARCH_SPACE),
                                         pruner=MultiPruner((optuna.pruners.ThresholdPruner(lower=0.02), optuna.pruners.HyperbandPruner())))
        
        if os.path.exists(f'{self.cfg.HP_SEARCH_DIR}/{self.name}.pkl'):
            study_old = joblib.load(f'{self.cfg.HP_SEARCH_DIR}/{self.name}.pkl')
            self.study.add_trials(study_old.get_trials())
            print('Study resumed!')
        
        save_callback = SaveCallback(self.cfg.HP_SEARCH_DIR)
        self.study.optimize(lambda trial: self.objective(trial), n_trials=self.cfg.TRIALS, callbacks=[save_callback])

        pruned_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])

        self.logger.save_log("Study statistics: ")
        self.logger.save_log(f"  Number of finished trials: {len(self.study.trials)}")
        self.logger.save_log(f"  Number of pruned trials: {len(pruned_trials)}")
        self.logger.save_log(f"  Number of complete trials: {len(complete_trials)}")
        self.logger.save_log("Best trial:")
        self.logger.save_log(f"  Value: {self.study.best_trial.value}")
        self.logger.save_log("  Params: ")
        for key, value in self.study.best_trial.params.items():
            self.logger.save_log(f"    {key}: {value}")

        return self.study
    

class SaveCallback:
    def __init__(self, directory):
        self.directory = directory
    def __call__(self, study, trial):
        joblib.dump(study, os.path.join(self.directory, f'{study.study_name}.pkl'))