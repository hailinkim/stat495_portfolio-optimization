import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_indicators(df, n=14):
    """
    Generates technical indicators
    """
    def sma(n):
        # Simple Moving Average (SMA)
        return df['Close'].rolling(window=n).mean()
    def wma(n):
        # Weighted Moving Average (WMA)
        weights = np.arange(1, n + 1)
        return df['Close'].rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    def ema(n):
        # Exponential Moving Average (EMA)
        return df['Close'].ewm(span=n, adjust=False).mean()

    # True Range
    df['TrueRange'] = np.maximum(df['High'] - df['Low'], 
                                 np.maximum(np.abs(df['High'] - df['Close'].shift(1)), 
                                            np.abs(df['Low'] - df['Close'].shift(1))))

    # Average True Range (ATR)
    df['ATR'] = df['TrueRange'].rolling(window=n, min_periods=1).mean()

    #Price rate of change
    df['PROC'] = (df['Close'] - df['Close'].shift(9)) / df['Close'].shift(9)
    #Stochastic Oscillator
    df['SO'] = ((df['Close'] - df['Low'].shift(14)) / (df['High'].shift(14) - df['Low'].shift(14)))*100
    #Williams Percent Range
    df['WPR'] = ((df['High'].shift(14) - df['Close']) / (df['High'].shift(14) - df['Low'].shift(14)))* (-100)

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    for i in [2,7, 9, 14, 30]:
        #Simple moving average
        df[f'SMA_{i}'] = sma(i)
        #Weighted moving average
        # df[f'WMA_{i}'] = wma(i)
    df['EMA_9'] = ema(9)
    # Moving Average Convergence Divergence (MACD)
    ema12 = ema(12)
    ema26 = ema(26)
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['BB_Upper'] = sma(20) + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = sma(20) - 2 * df['Close'].rolling(20).std()

    # On Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # fourier_transform = np.fft.fft(df['Close'])
    # df['fft_magnitude'] = np.abs(fourier_transform)
    # df['fft_phase'] = np.angle(fourier_transform)

    # # Average Directional Movement Index (ADX)
    # # (Calculating ADX is complex and needs several steps, including calculating DM+ and DM-, the ATR, and then the ADX)

    # # Commodity Channel Index (CCI)
    # TP = (df['High'] + df['Low'] + df['Close']) / 3
    # df['CCI'] = (TP - TP.rolling(n).mean()) / (0.015 * TP.rolling(n).std())

    # # Stochastic Momentum Index (SMI)
    # # (SMI is a more complex indicator and requires several calculations)

    # # Ease of Movement (EoM, EMV)
    # high_low = (df['High'] + df['Low']) / 2
    # move = high_low.diff()
    # box = (df['Volume'] / (df['High'] - df['Low']))
    # df['EoM'] = move / box

    # # Money Flow Index (MFI)
    # # (MFI calculation involves typical price and money flow, and then applying a ratio of positive and negative money flow)

    # # Chaikin Money Flow (CMF)
    # MFV = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    # df['CMF'] = MFV.rolling(n).sum() / df['Volume'].rolling(n).sum()

    # # Volume-Weighted Average Price (VWAP)
    # cum_vol = df['Volume'].cumsum()
    # cum_vol_price = (df['Close'] * df['Volume']).cumsum()
    # df['VWAP'] = cum_vol_price / cum_vol
    return df

def generate_data(ticker, period = '10y'):
    # df = pd.DataFrame()
    cols = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']
    # for ticker in tickers:
    tickerData = yf.Ticker(ticker)
    ticker_df = tickerData.history(period=period)
    # ticker_df['Ticker'] = ticker
    ticker_df = calculate_indicators(ticker_df)
    # shift the target variable up by 1 day to predict the next day's price
    ticker_df['Close'] = ticker_df['Close'].shift(-1)
    # add lagged target variable
    for i in [1,2]:
        ticker_df[f'Close_lag{i}'] = ticker_df['Close'].shift(i)
    ticker_df['Year'] = ticker_df.index.year
    ticker_df['Quarter'] = ticker_df.index.quarter
    ticker_df['Month'] = ticker_df.index.month
    ticker_df['Week'] = ticker_df.index.isocalendar().week 
    ticker_df['Day'] = ticker_df.index.day
    ticker_df['DayOfWeek'] = ticker_df.index.dayofweek
    ticker_df['DayOfYear'] = ticker_df.index.dayofyear
    # exclude unnecessary columns
    ticker_df.drop(columns=cols, inplace=True)
    # exclude NAs created from moving averages and shifting target variable
    ticker_df = ticker_df.dropna()#iloc[:-1,:]
    # check if missing values are removed properly
    assert not ticker_df.isna().any().any(), "missing values exist"
    # make sure the data is sorted by date
    ticker_df.sort_index(inplace=True)
    # df = pd.concat([df, ticker_df], axis=0)
    return ticker_df

def create_lag(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('%s_lag%d' % (df.columns[j], i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[["Close"]].shift(-i))
        if i == 0:
            names += ['Close']
        else:
            names += ['Close_lead%d' % i]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    # Make sure to return the DataFrame with the original index
    agg.index = df[n_in:].index
    
    # add time-related features
    agg['Year'] = agg.index.year
    agg['Quarter'] = agg.index.quarter
    agg['Month'] = agg.index.month
    agg['Week'] = agg.index.isocalendar().week 
    agg['Day'] = agg.index.day
    agg['DayOfWeek'] = agg.index.dayofweek
    agg['DayOfYear'] = agg.index.dayofyear
    
    return agg

def split_data(df, train_ratio, validation_ratio):
    
    if train_ratio + validation_ratio < 1:
        train = df[df.index < '2022-01-01']
        val = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]
        test = df[df.index >= '2023-01-01']
        X_train = train.drop(['Close'], axis=1)  # Features
        y_train = train['Close'].copy()  # Target
        X_val = val.drop(['Close'], axis=1)  # Features
        y_val = val['Close'].copy()  # Target
        X_test = test.drop(['Close'], axis=1)  # Features
        y_test = test['Close'].copy()  # Target
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    else:
        train = df[df.index < '2023-01-01']
        test = df[(df.index >= '2023-01-01')]
        X_train = train.drop(['Close'], axis=1)  # Features
        y_train = train['Close'].copy()  # Target
        X_test = test.drop(['Close'], axis=1)  # Features
        y_test = test['Close'].copy()  # Target
        return X_train, y_train, X_test, y_test

#     n = len(df)
#     train_end = int(n * train_ratio)
#     validation_end = int(n * (train_ratio + validation_ratio))

#     # Split the data
#     train = df[:train_end]
#     val = df[train_end:validation_end]
    
#     if train_ratio + validation_ratio < 1:
#         test = df[validation_end:]

    
    
    # if train_ratio + validation_ratio < 1:
    #     X_test = test.drop(['Close'], axis=1)  # Features
    #     y_test = test['Close'].copy()  # Target
    #     return X_train, y_train, X_val, y_val, X_test, y_test
    # else: 
    #     return X_train, y_train, X_val, y_val      
    

def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    # Apply transform to both the training set and the test set
    X_train = scaler.transform(X_train)
    # X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def plot_forecast(ticker, train_price_forecast,test_price_forecast, train, test):
    # Plotting the actual data
    plt.plot(train, color='orange')
    plt.plot(test, label = 'Actual', color = 'orange') #x=actual_data.index,
    # Plotting the predicted data
    sns.lineplot(y=train_price_forecast, x = train.index, label = 'Predicted (In-sample)', color = 'blue')
    sns.lineplot(y=test_price_forecast, x = test.index, label = 'Predicted (Out-of-sample)', color = 'green')
    
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.title(f'{ticker}')
    plt.legend()