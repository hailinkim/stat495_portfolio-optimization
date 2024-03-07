from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
import xgboost as xgb

def forecast(model, X_test, y_test):
    dtest = xgb.DMatrix(X_test, label=y_test)
    y_pred = model.predict(dtest)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    # print(f"RMSE: {rmse}, MAPE: {mape}")
    return y_pred, mape, rmse

