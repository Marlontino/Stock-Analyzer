import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf

# Choose timeframe
start = dt.datetime(2020,1,1)
end = dt.datetime(2023,12,31)
style.use('ggplot')

# download dataframe
yf.pdr_override() # <== that's all it takes :-)
appl = pdr.get_data_yahoo("AAPL", start, end)

# Read in csv
df = pd.read_csv('aapl.csv', parse_dates=True, index_col=0)

# Calculate moving average
df['100ma'] = df['Adj Close'].rolling(window=100).mean()
df.dropna(inplace=True)

# Plot ma
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()