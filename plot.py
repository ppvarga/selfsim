from locale import format_string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import csv
import datetime as dt
from sklearn.linear_model import LinearRegression
import random
import ruptures as rpt
import pandas as pd
import jenkspy


def remove_trends(df, do_plot):
    ts = df['value']

    breaks = pelt(ts)

    breaks_rpt = []
    notrend = np.array([])

    for i in range(len(breaks)-1):
        l = breaks[i]
        h = breaks[i+1]

        frame = df[l:h]
        
        frame['ordinal'] = frame.date.map(dt.datetime.toordinal)

        d = frame.date.values
        x = frame.ordinal.values.reshape(-1,1)
        y = frame.value.values.reshape(-1,1)

        regr = LinearRegression()
        regr.fit(x,y)

        p = regr.predict(x)

        notrend = np.concatenate((notrend, (y-p).reshape(-1)))

        if do_plot:
            plt.plot(d,p,c='grey')
            plt.scatter(d,y,s=1)
            breaks_rpt.append(ts.index[h-1])

    if do_plot:
        breaks_rpt = pd.to_datetime(breaks_rpt)

        for i in breaks_rpt:
            plt.axvline(i, color='red',linestyle='dashed')
        plt.grid()

        plt.subplot(1,2,2)
        plt.scatter(df.date.values, notrend, s=1)
        plt.show()

    data = {'date': df.date.values, 'value': notrend}

    return pd.DataFrame(data)

def parse_csv():
    df = pd.read_csv('data-deviza.csv')
    currency = input("Which currency would you like to see?")
    df['date'] = pd.to_datetime(df['date'])
    df = df[df["iso_4217"] == currency ]
    df = df[df['date'] > dt.datetime(2000,1,1)]
    df.set_index(df['date'], inplace = True)
    return df, currency

def pelt(ts):
    y0 = np.array(ts.tolist())

    breakpoint_model = rpt.Pelt(min_size=30,model="rbf")
    breakpoint_model.fit(y0)
    return [0] + breakpoint_model.predict(pen=10)

def plot_nth(df, title, div = 10, i = 0, j = 0):
    split_large = np.array_split(df, div)
    split = split_large[i].value.values
    nth = df.iloc[j::div, :].value.values

    fig, ((ax1, ax2), (hist1, hist2)) = plt.subplots(2,2, sharex = 'row', sharey = 'row')

    ax1.scatter(np.arange(len(split)), split, s = 2)
    ax1.title.set_text("First " + str(div) + "th of data")

    ax2.scatter(np.arange(len(nth)), nth, s = 2)
    ax2.title.set_text("Every " + str(div) + "th datapoint")

    hist1.hist(split, density=True, bins=10)
    hist1.title.set_text("\u03C3 = " + str(np.std(split)))

    hist2.hist(nth, density=True, bins =10)
    hist2.title.set_text("\u03C3 = " + str(np.std(nth)))
    
    fig.suptitle(title)
    plt.show()
    

df, title = parse_csv()
notrend = remove_trends(df, False)
plot_nth(notrend, title)
