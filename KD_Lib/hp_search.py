import os
import joblib
import optuna

from KD_Lib.train import train


class HPSearcher:
      
    def __init__(self, cfg, name, logger=None, trial=None):
        
        self.cfg = cfg
        self.name = name
        self.logger = logger
        self.trial = trial
        
        self.search_space = {"La": [1], # KLDiv Ts->S
                             "Lb": [1], # KLDiv S->Ts
                             "Lc": [0], # CE Ts [0, 0.1, 0.5, 1]
                             "Ld": [0, 0.1, 1], # MSE Ts->T
                             "Le": [0, 0.1, 0.5, 1], # KLDiv S->T
                             "Lf": [0], # SharpLoss
                             "T": [4], # Temperature
                             "W": [1], # KD
                            }
    
    def get_random_hps(self):
        self.cfg.T = self.trial.suggest_categorical("T", self.search_space["T"])
        self.cfg.W = self.trial.suggest_categorical("W", self.search_space["W"])
        self.cfg.L = [self.trial.suggest_categorical("La", self.search_space["La"]),
                      self.trial.suggest_categorical("Lb", self.search_space["Lb"]),
                      self.trial.suggest_categorical("Lc", self.search_space["Lc"]),
                      self.trial.suggest_categorical("Ld", self.search_space["Ld"]),
                      self.trial.suggest_categorical("Le", self.search_space["Le"]),
                      self.trial.suggest_categorical("Lf", self.search_space["Lf"])]
        print(f"T={self.cfg.T}, W={self.cfg.W}, L={self.cfg.L}")
    
    def objective(self, trial):
        self.cfg.EXP = f"{self.name}_{trial.number}"
        self.trial = trial 
        self.get_random_hps()

        metr = train(cfg=self.cfg, logger=self.logger, trial=self.trial)
        return metr
    
    
    def hp_search(self):
        self.study = optuna.create_study(study_name=self.name,
                                         direction='maximize', 
                                         sampler=optuna.samplers.GridSampler(self.search_space),
                                         pruner=optuna.pruners.HyperbandPruner())
        
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