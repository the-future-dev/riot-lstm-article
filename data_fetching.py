import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def new_stock(ticker):
    """
    Fetches a stock data using yahoo finance and saves it under its ticker name.
    """
    file_path = f'data/{ticker}.csv'
    data = yf.download(ticker, start=start, end=end)
    data.to_csv(file_path)

def fourier_transform(series, n_harmonics):
    detrended_series = series - series.mean()
    t = np.fft.fft(detrended_series)
    frequencies = np.fft.fftfreq(detrended_series.size)
    
    indices = np.argsort(np.abs(frequencies))
    t[indices[n_harmonics:]] = 0
    
    filtered_series = np.fft.ifft(t)
    return filtered_series.real + series.mean()

def smooth(series, window_size=5):
    return series.rolling(window=window_size).mean().fillna(0)

def technical_indicators_ta(ticker):
    """
    #1 Reads the stock data: assuming columns "Open" "Low" "High" "Close" "Adj Close" "Volume"
    #2 Creates various technical analysis indicators and various data.
    """
    stock_path = f'data/{ticker}.csv'
    file_path = f'data/{ticker}_ta.csv'
    df = pd.read_csv(stock_path, index_col=0, parse_dates=True)

    # Create 7 and 21 days Moving Average
    df['ma7'] = df['Close'].rolling(window=7).mean()
    df['ma7_diff'] = df['ma7'] - df['Close']
    df['ma21'] = df['Close'].rolling(window=21).mean()
    df['ma21_diff'] = df['ma21'] - df['Close']
    
    # Create MACD
    df['26ema'] = df['Close'].ewm(span=26).mean()
    df['26ema_diff'] = df['26ema'] - df['Close']
    df['12ema'] = df['Close'].ewm(span=12).mean()
    df['12ema_diff'] = df['12ema'] - df['Close']
    df['MACD'] = (df['12ema']-df['26ema'])

    # Create Bollinger Bands
    sd20 = df['Close'].rolling(window=20).std()
    df['upper_band'] = df['ma21'] + (sd20*2)
    df['lower_band'] = df['ma21'] - (sd20*2)
        
    # Create Momentum (using a period of 10 days for this example)
    df['momentum'] = df['Close'] - df['Close'].shift(10)

    smoothed = df['Close'].rolling(window=5).mean().fillna(0)
    df['fourier_short'] = fourier_transform(smoothed, 15)
    df['fourier_medium'] = fourier_transform(smoothed, 7)
    df['fourier_long'] = fourier_transform(smoothed, 3)

    df['Volatility_21'] = np.log(df['Close'] / df['Close'].shift(1)).rolling(window=21).std() * np.sqrt(21)
    df['OC diff'] = df['Open'] - df['Close']
    df['Close Diff'] = df['Close'].diff()
    df['Open Diff'] = df['Open'].diff()
    # df = df.drop('Adj Close', axis=1)

    df.to_csv(file_path)

if __name__ == "__main__":
    start = "2001-01-01"
    end = "2023-12-07"
    stock = "AZM.MI"

    new_stock(stock)
    technical_indicators_ta(stock)