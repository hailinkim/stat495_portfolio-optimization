from sklearn.ensemble import RandomForestRegressor
import os
from utils import generate_data, split_data, scale_data, plot_forecast, create_lag
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
from scipy.optimize import minimize
import json
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import numpy as np
import time
import datetime
import argparse
from sklearn.model_selection import GridSearchCV
from model.set_param_optuna import hyperparameter_tuning
from model.train import train
from model.predict import forecast
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import math

today = datetime.datetime.now().strftime("%Y%m%d")

class Asset:
    def __init__(self, ticker, data):
        """
        Initialize the Asset with historical data.
        :param data: DataFrame with historical data.
        """
        self.ticker = ticker
        self.data = data
        self.mape = 0
        self.predicted_price = None
        # self.predicted_daily_return = 0        
        
    def set_mape(self, predicted_price, mape):
        self.predicted_price = predicted_price
        self.mape = mape
        # print(f"MAPE updated successfully: {self.mape}")
        
    def history_daliy_return(self, log = True):
        """
        Calculate return based on the historical price.
        """
        if log:
            return np.log(self.data.iloc[1:].values / self.data.iloc[:-1]).values#.reshape(-1, 1)
        else: 
            return self.data.iloc[1:].values / self.data.iloc[:-1].values#.reshape(-1, 1)

    def predict_daliy_return(self, log = True):
        """
        Calculate return based on the predicted price.
        """
        if log:
            return np.log(self.predicted_price[1:] / self.predicted_price[:-1])#.reshape(-1, 1)
        else: 
            return self.predicted_price[1:] / self.predicted_price[:-1]#.reshape(-1, 1)

class Portfolio:
    FRQUENCY = 252  # Number of trading days in a year
    RISK_FREE_RATE = 0.0526  # 3-month T-bill rate (as of 12/8/23)
    
    def __init__(self, risk_tolerance=None, investment_horizon=None):
        self.assets = []
        self.weights = []
        self.risk_tolerance = risk_tolerance
        self.investment_horizon = investment_horizon

    def add_asset(self, asset, weight=None):
        self.assets.append(asset)
        if weight is not None:
            self.weights.append(weight)
    
    def select_top_n_assets(self, n):
        """
        Select top n assets based on the smallest MAPE.
        """
        sorted_assets = sorted(self.assets, key=lambda x: x.mape)
        self.assets = sorted_assets[:n]
        print("Selected assets: ", [asset.ticker for asset in self.assets])
    
    def daily_returns(self, forecast = True):
        """
        Store daily returns of individual assets in a portfolio in a list
        """
        daily_returns = []
        for asset in self.assets:
            if forecast:
                daily_return = asset.predict_daliy_return()
            else:
                daily_return = asset.history_daliy_return()
            daily_returns.append(daily_return)
        return np.array(daily_returns)

    def annualized_performance(self, weights, forecast=True):
        # annualized log return of the portfolio (multiply by number of trading days)
        portfolio_return = np.sum(np.mean(self.daily_returns(forecast), axis=1) * weights) * Portfolio.FRQUENCY
        # annualized log risk of the portfolio (multiply by number of trading days)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(np.cov(self.daily_returns(forecast)) * Portfolio.FRQUENCY, weights)))
        return portfolio_return, portfolio_risk
    
    def equal_weights(self):
        # generate equal weights for each asset
        self.weights = np.array([1/len(self.assets) for _ in range(len(self.assets))])
    
    def random_weights(self):
        # generate random weights for each asset
        weights = np.random.random(len(self.assets))
        weights /= np.sum(weights)           
        self.weights = weights              
    
    def simulate_portfolios(self, num_portfolios, forecast=True):
        # generate random portfolios to plot on the efficient frontier 
        results = np.zeros((num_portfolios, 3))
        for i in range(num_portfolios):
            # initialize random weights
            self.random_weights()
            portfolio_return, portfolio_risk = self.annualized_performance(self.weights, forecast)
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_risk
            results[i, 2] = (portfolio_return - Portfolio.RISK_FREE_RATE) / portfolio_risk
        return results
        
    def optimize_with_risk_tolerance(self, risk_tolerance, forecast=True):
        def objective_function(weights):
            portfolio_return, portfolio_risk = self.annualized_performance(weights, forecast)
            return portfolio_risk - risk_tolerance * portfolio_return
        
        weights = np.random.random(len(self.assets))
        weights /= np.sum(weights)
        bounds = [(0, 1) for _ in range(len(self.assets))]
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}]
        
        result = minimize(objective_function, weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            # update the weights of each asset after the optimization
            self.weights = result.x
        else:
            raise ValueError(f"Optimization failed: {result.message}")
            
    def optimize_sharpe_ratio(self, forecast=True):
        # Maximize Sharpe ratio = minimize minus Sharpe ratio
        def objective_function(weights):
            portfolio_return, portfolio_risk = self.annualized_performance(weights, forecast)
            sharpe_ratio = (portfolio_return - Portfolio.RISK_FREE_RATE) / portfolio_risk
            return -sharpe_ratio
        # initialize random weights
        self.random_weights()
        bounds = [(0, 1) for _ in range(len(self.assets))]
        constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}]

        # Add a risk tolerance constraint (currently not considered)
        if self.risk_tolerance is not None:
            risk_constraint = {'type': 'ineq', 'fun': lambda weights: portfolio_risk - self.risk_tolerance}
            constraints.append(risk_constraint)

        result = minimize(objective_function, self.weights, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            # update the weights of each asset after the optimization
            self.weights = result.x
        else:
            raise ValueError(f"Optimization failed: {result.message}")
            
    def create_efficient_frontier(self, forecast=True):
        # draw an efficient frontier (max return for every given risk)
        results = np.zeros((1000, 2))
        i=0
        for rt in np.linspace(-20, 30, 1000):
            self.optimize_with_risk_tolerance(rt, forecast)
            portfolio_return, portfolio_risk = self.annualized_performance(self.weights, forecast)
            results[i, 0] = portfolio_return
            results[i, 1] = portfolio_risk
            i+=1
        plt.figure(figsize=(10, 6))
        plt.plot(results[:,1], results[:, 0], 'k', linewidth=3, label='Efficient frontier')
        
        # Simulate 10000 random portfolios
        random_results = self.simulate_portfolios(10000)
        sharpe_ratio = random_results[:,2]
        scatter = sns.scatterplot(x=random_results[:, 1], y=random_results[:, 0], hue=sharpe_ratio, palette="viridis", 
                                  legend = False, edgecolor='gray')
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=np.min(sharpe_ratio), vmax=np.max(sharpe_ratio))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, label='Sharpe Ratio')
        
        # Generate portfolio with maximum sharpe ratio
        self.optimize_sharpe_ratio(forecast)
        portfolio_return, portfolio_risk = self.annualized_performance(self.weights, forecast)
        plt.plot(portfolio_risk, portfolio_return, 'rx', markeredgewidth=3, markersize=12, label='Maximum Sharpe Ratio (XGBoost)')  
        
        # Generate portfolio with minimum risk
        self.equal_weights()
        portfolio_return, portfolio_risk = self.annualized_performance(self.weights, forecast)
        plt.plot(portfolio_risk, portfolio_return, 'm+', markeredgewidth=3, markersize=12, label='Equal Weights')    
        
        # Generate portfolio with minimum risk
        self.optimize_with_risk_tolerance(0)
        portfolio_return, portfolio_risk = self.annualized_performance(self.weights, forecast)
        plt.plot(portfolio_risk, portfolio_return, 'b+', markeredgewidth=3, markersize=12, label='Minimum Risk')        
        
        plt.xlabel('Annualized Log Risk')
        plt.ylabel('Annualized Log Return')
        plt.title('Efficient Frontier')
        plt.legend()
        plt.savefig(f'plot/efficient_frontier_{today}.png')
        plt.show()
        
    def cumulative_return(self, test_df):
        assets = [asset.ticker for asset in portfolio.assets]
        # extract out-of-sample period
        start = test_df.index.min()
        end = test_df.index.max()
        price = yf.download(assets, start = start, end = end)['Close']  
        returns = price[assets].pct_change().dropna()
        
        weights = pd.Series(self.weights)
        weights.index = returns.columns
        portfolio_return = np.sum(returns * weights, axis=1)
        cum_return = (1 + portfolio_return).cumprod() - 1
        cum_return *= 100
        cum_return.name = 'cum_return'
        return cum_return        
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune_param', action = 'store_true', help = 'Tune hyperparameters')
    parser.add_argument('--cv', action = 'store_true', help = 'Cross-validation')
    parser.add_argument('--cv_fold', type = int, default = 5, help = 'Specify number of folds for CV')
    parser.add_argument('--seed', type = int, default = 11)
    parser.add_argument('--param_ver', help = 'Specify the version of tuned hyperparameters saved as JSON')
    parser.add_argument('--pretrained', action = 'store_true', help = 'Use a pretrained model')
    parser.add_argument('--forecast_mode', action='store_true', help = 'Use price forecasting for portfolio optimization (if False, same as Modern Portfolio Theory)')
    args = parser.parse_args()

    # set seed for reproducibility
    np.random.seed(args.seed) 
    
    # Set up subplots for price forecasting plot
    fig, axs = plt.subplots(5, 2, figsize=(20, 25)) # 5 rows, 2 columns
    i=0
    
    tickers = ['NVDA','JPM', 'JNJ', 'MSFT', 'XOM', 'AMZN', 'UNH', 'BRK-B', 'AAPL'] 
    results = []    
    rf_results = []
    portfolio = Portfolio()
    #create a folder for saving hyperparameters
    os.makedirs(f"./config/{today}/param", exist_ok=True)
    for ticker in tickers:
        print(f"{ticker} started")
        historical_prices = generate_data(ticker, '10y')
        # historical_prices = create_lag(historical_prices, n_in=7)
            
        if args.tune_param:
            base_param = f'config/{today}/param/params_{ticker}_{today}_{args.cv}'
            file_index = 0
            param_file = f"{base_param}_scaled.json"

            # while os.path.exists(param_file):
                # file_index += 1
            #     param_file = f"{base_param}_{file_index}_scaled.json"   

            if args.cv:
                X_train, y_train, X_test, y_test = split_data(historical_prices, 0.8, 0.2)  
                if ticker == "NVDA":
                    print("full data: ", historical_prices.shape)
                    print(y_train.index.min(), y_train.index.max())
                    print(y_test.index.min(), y_test.index.max())
                X_train, _ = scale_data(X_train, X_test)
                hyperparameter_tuning(ticker, X_train = X_train, y_train = y_train, 
                                      param_file = param_file, cv = args.cv, cv_fold = args.cv_fold, seed = args.seed)
            else:
                X_train, y_train, X_val, y_val, X_test, y_test = split_data(historical_prices, 0.8, 0.1)   
                hyperparameter_tuning(ticker, X_train = X_train, y_train = y_train, X_val = X_val, y_val=y_val,
                                      param_file = param_file, cv = args.cv, seed = args.seed)
        else:
            if args.cv:
                X_train, y_train, X_test, y_test = split_data(historical_prices, 0.8, 0.2)       
                if ticker == "NVDA":
                    print("full data: ", historical_prices.shape)
                    print(y_train.index.min(), y_train.index.max())
                    print(y_test.index.min(), y_test.index.max())
            else:     
                X_train, y_train, X_val, y_val, X_test, y_test = split_data(historical_prices, 0.8, 0.1)   
                X_train = pd.concat([X_train, X_val], axis=0)
                y_train = pd.concat([y_train, y_val], axis=0)
                
            X_train, X_test = scale_data(X_train, X_test)
            model = train(ticker, X_train, y_train, param_ver = args.param_ver, cv = args.cv, save_model = True)
                
            print("forecast started")   
            # in-sample prediction (to plot the predictions)
            train_price_forecast, _, _ = forecast(model, X_train, y_train)
            # out-of-sample prediction
            test_price_forecast, mape, rmse = forecast(model, X_test, y_test)
            print(f'XGB - RMSE: {rmse}, MAPE: {mape}')

            asset = Asset(ticker, y_train)
            asset.set_mape(test_price_forecast, mape)
            portfolio.add_asset(asset)

            # Plot the forecasts
            row = i // 2  # Integer division to determine the row index
            col = i % 2  # Remainder to determine the column index
            plt.sca(axs[row, col])
            results.append({'Ticker': ticker, 'MAPE': mape, 'RMSE': rmse})
            plot_forecast(ticker, train_price_forecast, test_price_forecast, y_train, y_test)
            i+=1
            
            rf_model = RandomForestRegressor(n_estimators=100, max_depth = 10, random_state=42) # You can tune these parameters

            # Use these scaled datasets for training and predictions
            rf_model.fit(X_train, y_train)
            y_pred_rf = rf_model.predict(X_test)

#             plot_forecast(ticker, y_pred_train, y_pred_test, y_train, y_test)
#             plt.show()

            # Evaluate the model
            rf_rmse = mean_squared_error(y_test, y_pred_rf, squared=False)
            rf_mape = mean_absolute_percentage_error(y_test, y_pred_rf)
            rf_results.append({'Ticker': ticker, 'MAPE': rf_mape, 'RMSE': rf_rmse})
            
            
    if not args.tune_param:
        fig.suptitle('Predicted vs Actual Stock Prices', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'plot/stock_forecasting_{today}.png') 
        plt.show()

        # Save the out-of-sample prediction performance
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'plot/model_performance_{today}_XGBoost.csv', index=False)
        
        rf_results_df = pd.DataFrame(rf_results)
        rf_results_df.to_csv(f'plot/model_performance_{today}_RF.csv', index=False)

        # add top 5 assets to portfolio with lowest MAPE
        # portfolio.select_top_n_assets(5)
        # draw and save efficient frontier
        portfolio.create_efficient_frontier()

        performance = []
        # sharpe ratio optimization
        portfolio.optimize_sharpe_ratio()
        portfolio_return, portfolio_risk = portfolio.annualized_performance(portfolio.weights)
        sharpe_ratio = (portfolio_return - portfolio.RISK_FREE_RATE) / portfolio_risk
        performance.append({'Strategy': 'XGBoost_Sharpe', 
                            'Annual Return': math.exp(portfolio_return)-1, 
                            'Annual Risk': portfolio_risk, 
                            'Annual Sharpe Ratio': sharpe_ratio})
        assets = [asset.ticker for asset in portfolio.assets]
        weights = pd.DataFrame(list(zip(assets, portfolio.weights)), columns=['Ticker', 'Weight'])
        weights.to_csv(f'plot/portfolio_weights_XGBoost_{today}.csv', index=False)

        # backtesting
        sharpe_cum_return = portfolio.cumulative_return(y_test)

        # equal weights portfolio
        portfolio.equal_weights()
        portfolio_return, portfolio_risk = portfolio.annualized_performance(portfolio.weights)
        sharpe_ratio = (portfolio_return - portfolio.RISK_FREE_RATE) / portfolio_risk
        performance.append({'Strategy': 'Equal Weights', 
                            'Annual Return': math.exp(portfolio_return)-1, 
                            'Annual Risk': portfolio_risk, 
                            'Annual Sharpe Ratio': sharpe_ratio})
        # backtesting
        equal_cum_return = portfolio.cumulative_return(y_test)

        ## history data
        # sharpe ratio optimization
        portfolio.optimize_sharpe_ratio(False)
        portfolio_return, portfolio_risk = portfolio.annualized_performance(portfolio.weights, False)
        sharpe_ratio = (portfolio_return - portfolio.RISK_FREE_RATE) / portfolio_risk
        performance.append({'Strategy': 'MPT_Sharpe', 
                            'Annual Return': math.exp(portfolio_return)-1, 
                            'Annual Risk': portfolio_risk, 
                            'Annual Sharpe Ratio': sharpe_ratio})
        assets = [asset.ticker for asset in portfolio.assets]
        weights = pd.DataFrame(list(zip(assets, portfolio.weights)), columns=['Ticker', 'Weight'])
        weights.to_csv(f'plot/portfolio_weights_MPT_{today}.csv', index=False)

        # backtesting
        past_sharpe_cum_return = portfolio.cumulative_return(y_test)

        #save portfolio performance for all three strategy
        performance_df = pd.DataFrame(performance)
        performance_df.to_csv(f'plot/portfolio_performance_{today}.csv', index=False)

        # Plot cumulative returns over time
        plt.figure(figsize=(10,6))
        plt.plot(pd.to_datetime(sharpe_cum_return.index), sharpe_cum_return, color='tab:orange', label = 'Sharpe ratio optimization (XGBoost)')
        plt.plot(pd.to_datetime(past_sharpe_cum_return.index), past_sharpe_cum_return, color='tab:red', label = 'Sharpe ratio optimization (History data)')
        plt.plot(pd.to_datetime(equal_cum_return.index), equal_cum_return, color='tab:cyan', label = 'Equal weights')
        plt.ylabel('Cumulative return (%)')
        plt.title('Portfolio Cumulative Return Backtesting (Out-of-Sample)', size = 16)
        plt.legend()
        plt.xticks(rotation=45)
        plt.savefig(f'plot/cumulative_return_{today}.png')
        plt.show()