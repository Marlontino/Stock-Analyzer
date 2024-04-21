import bs4 as bs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import requests
import datetime as dt
import pandas as pd
import pandas_datareader.data as pdr
import os
import yfinance as yf



def main():
    #save_sp500_tickers()
    #get_data()
    #compile_data()
    visualize_data()

def save_sp500_tickers():
    # Parse HTML
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.content, 'lxml')
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.find_all('tr')[1:]:
        ticker = row.find_all('td')[0].text
        tickers.append(ticker)

    # Remove \n from each elem
    tickers = [ticker.rstrip() for ticker in tickers]

    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    print(tickers)

    return tickers

def get_data(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open('sp500tickers.pickle', 'rb') as f:
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000,1,1)
    end = dt.datetime(2023,12,31)

    for ticker in tickers:
        ticker_file_path = 'stock_dfs/{}.csv'.format(ticker)
        if not os.path.exists(ticker_file_path):  # Check file existence for each ticker
            yf.pdr_override()
            df = pdr.get_data_yahoo(ticker, start, end)
            df.to_csv(ticker_file_path)
            print(f"Data saved for {ticker}")
        else:
            print(f'Already have data for {ticker}')

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame
    print(tickers[:5])

    for count, ticker in enumerate(tickers):
        df = pd.read_csv(f'stock_dfs/{ticker}.csv')
        df.set_index('Date', inplace=True)

        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joined.csv')

def visualize_data():
    # Read in data
    df = pd.read_csv('sp500_joined.csv', index_col=0, parse_dates=True)
    #df['AAPL'].plot()
    #plt.show()
    df_corr = df.corr()
    print(df_corr.head())

    
    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # Plot heatmap
    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = df_corr.columns
    row_labels = df_corr.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
