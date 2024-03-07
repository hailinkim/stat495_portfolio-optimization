import xgboost as xgb
from sklearn.metrics import mean_squared_error
import optuna
import logging
import sys
import datetime
import json
from optuna.integration import XGBoostPruningCallback
from optuna.integration import OptunaSearchCV
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

today = datetime.datetime.now().strftime("%Y%m%d")

def hyperparameter_tuning(ticker, X_train, y_train, param_file, X_val = None, y_val=None, cv=True, cv_fold=5, seed=42, save_results=True):
#     class StopWhenTrialKeepBeingPrunedCallback:
#         def __init__(self, threshold: int):
#             self.threshold = threshold
#             self._consequtive_pruned_count = 0

#         def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
#             if trial.state == optuna.trial.TrialState.PRUNED:
#                 self._consequtive_pruned_count += 1
#             else:
#                 self._consequtive_pruned_count = 0

#             if self._consequtive_pruned_count >= self.threshold:
#                 study.stop()
    def loss_callback(study, trial):
        intermediate_losses = trial.intermediate_values
        trial_number = trial.number

        elapsed_time = time.time() - study._storage.get_study_direction(study.study_id).study_user_attr['start_time']
        trial_number = trial.number

        # Print or log the intermediate loss values
        print(f"Trial {trial_number}: Intermediate Losses -> {intermediate_losses}, Elapsed Time: {elapsed_time:.2f} seconds")
    
    def objective_xgb(trial):
        # Suggest hyperparameters for XGBoost
        params = {'objective': 'reg:squarederror', #loss function
                  'eval_metric': 'rmse',
                  'seed': seed,
                  'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log = True),
                  'max_depth': trial.suggest_int('max_depth', 3, 20),
                  # 'num_parallel_tree': trial.suggest_int('num_parallel_tree', 100, 500),
                  'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                  'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                  'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
                  'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                  'gamma': trial.suggest_float('gamma', 0, 100),
                  'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 100, log=True),
                  'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 100, log=True),
                  "min_child_weight": trial.suggest_float("min_child_weight", 10, 1000, log=True),
                  'tree_method': 'hist',
                  'sampling_method': 'gradient_based',
                  'device': 'cuda:0'
                 }
        
        if cv:
            indices = []
            tscv = TimeSeriesSplit(n_splits=cv_fold, test_size=252)
            for train_index, test_index in tscv.split(X_train):
                indices.append((list(train_index), list(test_index)))
            dmat_train = xgb.DMatrix(X_train, label = y_train)
            pruning_callback = XGBoostPruningCallback(trial, 'test-rmse')    
            res = xgb.cv(params,
                        dmat_train,
                        num_boost_round=trial.suggest_int("num_boost_round", 3000, 10000),
                        # num_boost_round=trial.suggest_int("num_boost_round", 5, 20),
                        nfold=cv_fold,
                        folds = indices,
                        early_stopping_rounds = 50,
                        metrics={"rmse"},
                        seed=seed, 
                        callbacks = [pruning_callback]
                    )
            mean_rmse = res['test-rmse-mean'].values[-1]
            return mean_rmse
        else:
            dmat_train = xgb.DMatrix(X_train, label = y_train)
            dmat_valid = xgb.DMatrix(X_val, label = y_val)
            evallist = [(dmat_train, 'train'), (dmat_valid, 'validate')]
            pruning_callback = XGBoostPruningCallback(trial, 'validate-rmse')    
            xgb_model = xgb.train(params, 
                                  dtrain = dmat_train, 
                                  num_boost_round = trial.suggest_int("num_boost_round", 3000, 10000),
                                  early_stopping_rounds=50, 
                                  callbacks=[pruning_callback],
                                  evals = evallist)

            xgb_preds_valid = xgb_model.predict(dmat_valid) 
            return mean_squared_error(y_val, xgb_preds_valid, squared=False)


    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    # study_stop_cb = StopWhenTrialKeepBeingPrunedCallback(4)
    
    # Set up and run the Optuna study
    # pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction = 'minimize')
    study.optimize(objective_xgb, n_trials = 100, n_jobs=-1) #, callbacks=[loss_callback]
    
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
        
    # save best param
    # if save_results:
    with open(param_file, 'w') as file:
        json.dump(best_params, file)
    
    df = study.trials_dataframe()
    df.to_csv(f"./config/{today}/param/optuna_results_{ticker}_{cv}.csv", index=False)