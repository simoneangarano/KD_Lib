import os
import joblib

import optuna
from ray import tune
import ray.train
from ray.tune.search.optuna import OptunaSearch

from KD_Lib.train import train


class HPSearcher:
      
    def __init__(self, cfg, name, logger=None, trial=None):
        
        self.cfg = cfg
        self.name = name
        self.logger = logger
        self.trial = trial
            
    def set_random_hps(self):
        self.cfg.T = self.trial.suggest_categorical("T", self.cfg.SEARCH_SPACE["T"])
        self.cfg.W = self.trial.suggest_categorical("W", self.cfg.SEARCH_SPACE["W"])
        self.cfg.L = [self.trial.suggest_categorical("La", self.cfg.SEARCH_SPACE["La"]),
                      self.trial.suggest_categorical("Lb", self.cfg.SEARCH_SPACE["Lb"]),
                      self.trial.suggest_categorical("Lc", self.cfg.SEARCH_SPACE["Lc"]),
                      self.trial.suggest_categorical("Ld", self.cfg.SEARCH_SPACE["Ld"]),
                      self.trial.suggest_categorical("Le", self.cfg.SEARCH_SPACE["Le"]),
                      self.trial.suggest_categorical("Lf", self.cfg.SEARCH_SPACE["Lf"])]
        self.logger.save_log(f"T={self.cfg.T}, W={self.cfg.W}, L={self.cfg.L}")
    
    def objective(self, trial):
        self.cfg.EXP = f"{self.name}_{trial.number}"
        self.cfg.reset()
        trial.set_user_attr('name', self.cfg.EXP)
        self.trial = trial 
        self.set_random_hps()

        metr = train(cfg=self.cfg, logger=self.logger, trial=self.trial)
        return metr
    
    def hp_search(self):
        self.study = optuna.create_study(study_name=self.name,
                                         direction='maximize', 
                                         sampler=optuna.samplers.TPESampler(),
                                         pruner=optuna.pruners.ThresholdPruner(lower=0.02))
        
        if os.path.exists(f'{self.cfg.HP_SEARCH_DIR}/{self.name}.pkl'):
            study_old = joblib.load(f'{self.cfg.HP_SEARCH_DIR}/{self.name}.pkl')
            self.study.add_trials(study_old.get_trials())
            self.logger.save_log('Study resumed!')
        
        save_callback = SaveCallback(self.cfg.HP_SEARCH_DIR)
        self.study.optimize(lambda trial: self.objective(trial), n_trials=self.cfg.N_TRIALS, callbacks=[save_callback])

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
    
class HPTuner(HPSearcher):
    def __init__(self, cfg, name, logger=None, trial=None):
        super().__init__(cfg, name, logger, trial)
        ray.init()
        search_space = {"T": tune.choice(self.cfg.SEARCH_SPACE["T"]),
                        "W": tune.choice(self.cfg.SEARCH_SPACE["W"]),
                        "La": tune.choice(self.cfg.SEARCH_SPACE["La"]),
                        "Lb": tune.choice(self.cfg.SEARCH_SPACE["Lb"]),
                        "Lc": tune.choice(self.cfg.SEARCH_SPACE["Lc"]),
                        "Ld": tune.choice(self.cfg.SEARCH_SPACE["Ld"]),
                        "Le": tune.choice(self.cfg.SEARCH_SPACE["Le"]),
                        "Lf": tune.choice(self.cfg.SEARCH_SPACE["Lf"])}
        algorithm = OptunaSearch(metric="accuracy", mode="max", 
                                 sampler=optuna.samplers.TPESampler())
        tune_config = tune.TuneConfig(metric="accuracy", mode="max",
                                      search_alg=algorithm, num_samples=cfg.N_TRIALS,
                                      trial_name_creator=lambda trial: self.trial_str_creator(trial))
        run_config = ray.train.RunConfig(name=self.name, storage_path=self.cfg.HP_SEARCH_DIR)
        self.tuner = tune.Tuner(trainable=tune.with_resources(self.objective, {"gpu": 1}),
                                tune_config=tune_config, run_config=run_config, param_space=search_space)

    def set_random_hps(self, config):
        self.cfg.T = config["T"]
        self.cfg.W = config["W"]
        self.cfg.L = [config["La"], config["Lb"], config["Lc"], config["Ld"], config["Le"], config["Lf"]]
        self.logger.save_log(f"T={self.cfg.T}, W={self.cfg.W}, L={self.cfg.L}")

    def objective(self, config):
        self.cfg.EXP = f"{self.name}_{self.cfg.TRIAL}"
        self.cfg.reset()
        self.set_random_hps(config)
        metr = train(cfg=self.cfg, logger=self.logger, trial=self.trial)
        return {"accuracy": metr}
    
    def hp_search(self):
        results = self.tuner.fit()
        self.logger.save_log(f"Best config is: {results.get_best_result().config}")

    def trial_str_creator(self, trial):
        return f"{self.name}_{trial.trial_id}"
        

class SaveCallback:
    def __init__(self, directory):
        self.directory = directory
    def __call__(self, study, trial):
        joblib.dump(study, os.path.join(self.directory, f'{study.study_name}.pkl'))