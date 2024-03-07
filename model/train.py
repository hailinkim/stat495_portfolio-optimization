from xgboost import XGBRegressor
import pickle, os, json
import datetime
import xgboost as xgb    

def train(ticker, X_train, y_train, param_ver=None, cv = True, seed=11, save_model = True):
    # load saved hyperparameters tuned from grid search
    if param_ver is not None:     
        param = f'config/{param_ver}/param/params_{ticker}_{param_ver}_{cv}.json'
        with open(param, 'r') as file:
            best_params = json.load(file)

    best_params['tree_method'] = 'hist'
    best_params['sampling_method'] = 'gradient_based'
    best_params['device'] = 'cuda:0'
    # best_params['eval_metric'] = 'rmse'
            
    # convert data into dMatrix required by xgboost for efficient training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    # dvalid = xgb.DMatrix(X_val, label=y_val)
    # train model
    ## num_boost_round = number of trees
    # evallist = [(dtrain, 'train'), (dvalid, 'validate')]
    bst = xgb.train(best_params, 
                    dtrain, 
                    num_boost_round = best_params["num_boost_round"]) 
                    # early_stopping_rounds = 50,
                    # evals = evallist
    
    score = bst.get_score(importance_type='gain')
    print(score)
    # if save_model:
    #     today = datetime.datetime.now().strftime("%Y%m%d")
    #     os.makedirs(f'config/{param_ver}/model', exist_ok=True)
    #     bst.save_model(f'config/{param_ver}/model/model_{ticker}_{today}.json')
    return bst