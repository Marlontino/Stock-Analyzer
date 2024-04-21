from collections import Counter
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function to process data for labels
def process_data_for_labels(ticker):
    hm_days = 7
    # Read the dataset
    df = pd.read_csv('sp500_joined.csv', index_col=0)
    # Get column names
    tickers = df.columns.values.tolist()
    # Fill NaN values with 0
    df.fillna(0, inplace=True)

    # Calculate future changes in stock prices
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    # Fill NaN values with 0 again
    df.fillna(0, inplace=True)
    return tickers, df

# Function to determine whether to buy, sell, or hold
def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1  # Buy
        if col < -requirement:
            return -1  # Sell
    return 0  # Hold

# Function to extract featuresets
def extract_featuresets(ticker):
    # Process data for labels
    tickers, df = process_data_for_labels(ticker)

    # Create target column
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    # Get values and counter
    vals = df['{}_target'.format(ticker)].values.tolist()
    print('Data spread:', Counter(vals))

    # Handle NaN and inf values
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    # Calculate percentage change and handle NaN and inf values
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    # Features and labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df

# Function to perform machine learning
def do_ml(ticker):
    # Extract featuresets
    X, y, _ = extract_featuresets(ticker)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Create a voting classifier
    clf = VotingClassifier([('lsvc', LinearSVC()),
                            ('knn', KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    # Fit the model
    clf.fit(X_train, y_train)
    # Calculate accuracy
    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)
    # Make predictions
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))

    return confidence

# Test the function
do_ml('BAC')
