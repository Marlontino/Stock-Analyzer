import numpy as np 
import pandas as pd
import pickle

def main():
    pass

def process_data_for_labels(ticker):
    days = 7
    df = pd.read_csv('sp500_joined.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    # Calculate % change
    for i in range(1, days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker])
    
    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    pass

if __name__ == '__main__':
    main()