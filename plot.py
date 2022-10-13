from locale import format_string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from sklearn.linear_model import LinearRegression
import ruptures as rpt
import pandas as pd
import statsmodels.api as sm
import parse


def remove_trends(df, do_plot):
    ts = df['value']

    breaks = pelt(ts)

    breaks_rpt = []
    notrend = np.array([])
    
    if do_plot:
        fig, (ax1, ax2) = plt.subplots(1,2)

    for i in range(len(breaks)-1):
        l = breaks[i]
        h = breaks[i+1]

        frame = df[l:h]

        d = frame.date.values
        x = frame.ordinal.values.reshape(-1,1)
        y = frame.value.values.reshape(-1,1)

        regr = LinearRegression()
        regr.fit(x,y)

        p = regr.predict(x)

        notrend = np.concatenate((notrend, (y-p).reshape(-1)))

        if(do_plot):
            ax1.plot(d,p,c='grey')
            ax1.scatter(d,y,s=1)
            breaks_rpt.append(ts.index[h-1])

    if(do_plot):
        breaks_rpt = pd.to_datetime(breaks_rpt)

        for i in breaks_rpt:
            ax1.axvline(i, color='red',linestyle='dashed')
        ax1.grid()

        ax2.scatter(df.date.values, notrend, s=1)

    data = {'date': df.date.values, 'value': notrend}

    return pd.DataFrame(data)

def parse_curr_csv(currency):
    df = pd.read_csv('data-deviza.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df["iso_4217"] == currency ]
    df = df[df['date'] > dt.datetime(2000,1,1)]
    df.set_index(df['date'], inplace = True)
    df['ordinal'] = df.date.map(dt.datetime.toordinal)
    return df

def pelt(ts):
    y0 = np.array(ts.tolist())

    breakpoint_model = rpt.Pelt(min_size=30,model="rbf")
    breakpoint_model.fit(y0)
    return [0] + breakpoint_model.predict(pen=10)

def plot_nth(df, div = 10, i = 0, j = 0):
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

def plot_autocorrelation(values, separate, label, max_delay=50):
    acorr = sm.tsa.acf(values, nlags=max_delay)
    if separate:
        fig, ax = plt.subplots()
        ax.plot(range(max_delay+1), acorr)
        fig.suptitle("Autocorrelation as a function of the delay")
    else:
        plt.plot(range(max_delay+1), acorr, label=label)
        plt.title("Autocorrelations as functions of the delay")

def plot_one_currency():
    currency = input("Which currency would you like to see?")
    df = parse_curr_csv(currency)
    notrend = remove_trends(df, True)
    plot_nth(notrend)
    plot_autocorrelation(notrend.value)

def compare_autocorrs(max_delay):
    fig, ax = plt.subplots()
    for currency in ["USD", "CAD", "JPY", "CHF", "GBP"]:
        df = parse_curr_csv(currency)
        notrend = remove_trends(df, False)
        plot_autocorrelation(notrend.value, False, currency, max_delay=max_delay)
    ax.legend()

def parse_xz(file_id = 0, dataset_size=50000):
    filename = "xz{id}.csv".format(id=file_id)
    df = pd.read_csv(filename, nrows=dataset_size)
    df.rename(columns={"Signal Unit Time Stamp (hi-res)": "timestamp"}, inplace=True)
    format_str = "00:00:{sec}.{mili}'{micro}'{p}"
    
    nums = np.zeros(dataset_size)
    for i in range(dataset_size):
        parsed = parse.parse(format_str, df["timestamp"][i])
        nums[i] = int(str(parsed["sec"]) + str(parsed["mili"]) + str(parsed["micro"]) + str(parsed["p"]))
    df["nums"] = nums

    df["diffs"] = np.concatenate(( np.array([0]), np.diff(nums)))

    return df

def plot_xz_autocorr(file_id = 0, dataset_size = 40000, max_delay = 15):
    df = parse_xz(file_id, dataset_size)
    fig, (ax1, ax2) = plt.subplots(1,2)
    x = np.arange(dataset_size)
    ax1.scatter(np.arange(dataset_size), df.diffs)
    acorr = sm.tsa.acf(df.diffs, nlags=max_delay)
    ax2.plot(np.arange(max_delay+1), acorr)
for currency in ['USD', 'CAD', 'JPY', 'GBP', 'CHF']:
    trendy = parse_curr_csv(currency)
    notrend = remove_trends(trendy, False)
    notrend.to_csv("{curr}_notrend.csv".format(curr =currency), index=False)
