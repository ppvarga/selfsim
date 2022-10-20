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
import hurst

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
            ax1.axvline(i, color='red',linestyle='dashed', linewidth=1)
        ax1.grid()

        ax2.scatter(df.date.values, notrend, s=1)

    data = {'date': df.date.values, 'value': notrend}

    return pd.DataFrame(data)

def parse_curr_csv(currency):
    df = pd.read_csv('data-deviza.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df["iso_4217"] == currency ]
    df = df[df['date'] >= dt.datetime(2000,1,1)]
    df.set_index(df['date'], inplace = True)
    df['ordinal'] = df.date.map(dt.datetime.toordinal)
    return df

def pelt(ts):
    y0 = np.array(ts.tolist())

    breakpoint_model = rpt.Pelt(min_size=30,model="rbf")
    breakpoint_model.fit(y0)
    return [0] + breakpoint_model.predict(pen=10)

def plot_nth(df, div = 10, i = 0, j = 0, max_delay = 250):
    split_large = np.array_split(df, div)
    split = split_large[i].value.values
    nth = df.iloc[j::div, :].value.values

    fig, ((ax1, ax2), (hist1, hist2), (acorr1, acorr2)) = plt.subplots(3,2, sharex = 'row', sharey = 'row')

    ax1.scatter(np.arange(len(split)), split, s = 2)
    ax1.title.set_text("First " + str(div) + "th of data")

    ax2.scatter(np.arange(len(nth)), nth, s = 2)
    ax2.title.set_text("Every " + str(div) + "th datapoint")

    hist1.hist(split, density=True, bins=10)
    hist1.title.set_text("\u03C3 = " + str(np.std(split)))

    hist2.hist(nth, density=True, bins =10)
    hist2.title.set_text("\u03C3 = " + str(np.std(nth)))

    acorr_split = sm.tsa.acf(split, nlags=max_delay)
    acorr1.plot(range(max_delay+1), acorr_split)

    acorr_nth = sm.tsa.acf(nth, nlags=max_delay)
    acorr2.plot(range(max_delay+1), acorr_nth)

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

def parse_xz(file_id = 0, dataset_size=40000, compute_diffs = True):
    filename = "xz{id}.csv".format(id=file_id)
    df = pd.read_csv(filename, nrows=dataset_size)
    df.rename(columns={"Signal Unit Time Stamp (hi-res)": "timestamp"}, inplace=True)
    format_str = "00:00:{sec}.{mili}'{micro}'{p}"
    
    nums = np.zeros(dataset_size)
    for i in range(dataset_size):
        parsed = parse.parse(format_str, df["timestamp"][i])
        nums[i] = int(str(parsed["sec"]) + str(parsed["mili"]) + str(parsed["micro"]) + str(parsed["p"]))
    df["nums"] = nums

    if(compute_diffs):
        df["diffs"] = np.concatenate(( np.array([0]), np.diff(nums)))

    return df

def plot_xz_autocorr(file_id = 0, dataset_size = 40000, max_delay = 15, only_autocorr = False):
    df = parse_xz(file_id, dataset_size)
    x = np.arange(dataset_size)
    acorr = sm.tsa.acf(df.diffs, nlags=max_delay)

    if(only_autocorr):
        plt.plot(np.arange(max_delay+1), acorr)
    else:
        fig, (ax1, ax2) = plt.subplots(1,2)
        ax1.scatter(np.arange(dataset_size), df.diffs)
        ax2.plot(np.arange(max_delay+1), acorr)

def plot_water_level_acorr(max_delay = 2000):
    df = pd.read_csv('data-vizallas.csv')
    df['date'] = pd.to_datetime(df['date'])
    counter = 0
    locs = df.individual.unique()
    #fig, axs = plt.subplots(len(locs))
    figacr, axsacr = plt.subplots()
    #figacr.suptitle("Autocorrelations as functions of the delay")
    for loc in locs:
        locdf = df[df["individual"] == loc ]
        locdf.set_index(locdf['date'], inplace = True)
        locdf['ordinal'] = locdf.date.map(dt.datetime.toordinal)
        #axs[counter].scatter(locdf.ordinal, locdf.value, s=1)
        #axs[counter].title.set_text(loc)
        acorr = sm.tsa.acf(locdf.value, nlags = max_delay)
        axsacr.plot(np.arange(max_delay+1), acorr, label=loc)
        counter += 1
    axsacr.legend()

def hurst_curr():
    print("Hurst exponents for currencies")
    for curr in ['USD', 'CAD', 'JPY', 'CHF', 'GBP']:
        df = parse_curr_csv(curr)
        Htrend, c, data = hurst.compute_Hc(df.value, kind='random_walk', simplified=True)
        notrend = remove_trends(df, False)
        Hnotrend, c, data = hurst.compute_Hc(notrend.value, kind='random_walk', simplified=True)
        print("{curr}: {Htrend} with trend, {Hnotrend} with no trend".format(curr = curr, Htrend = Htrend, Hnotrend = Hnotrend))

def hurst_water_levels():
    print("Hurst exponents for water levels")
    df = pd.read_csv('data-vizallas.csv')
    df['date'] = pd.to_datetime(df['date'])
    locs = df.individual.unique()
    for loc in locs:
        locdf = df[df["individual"] == loc ]
        H, c, data = hurst.compute_Hc(locdf.value, kind='random_walk', simplified=True)
        print("{loc}: {H}".format(loc = loc, H = H))

def hurst_xz(file_id = 0, dataset_size = 40000, plot_normalized = False):
    df = parse_xz(file_id, dataset_size, True)
    H, c, data = hurst.compute_Hc(df.diffs, kind='random_walk', simplified=True)
    print("Hurst exponent for the diffs of the first {n} datapoints of xz{id}.csv: {H}".format(n = dataset_size, id = file_id, H = H))

    normalized_diffs = df.diffs - np.mean(df.diffs)
    Hnorm, c, data = hurst.compute_Hc(normalized_diffs, kind='change', simplified=True)
    print("Hurst exponent for the normalized diffs of the first {n} datapoints of xz{id}.csv: {H}".format(n = dataset_size, id = file_id, H = Hnorm))
    if(plot_normalized):
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.scatter(np.arange(dataset_size), normalized_diffs, s=1)
        ax2.plot(np.arange(dataset_size), np.cumsum(normalized_diffs))
        fig, ax = plt.subplots(1)
        max_delay = 100
        acorr = sm.tsa.acf(normalized_diffs, nlags = max_delay)
        ax.plot(np.arange(max_delay+1), acorr)

    diff_diffs = np.concatenate((np.array([0]), np.diff(df.diffs)))
    Hdiff, c, data = hurst.compute_Hc(diff_diffs, kind='change', simplified=True)
    print("Hurst exponent for the diffs of the diffs of the first {n} datapoints of xz{id}.csv: {H}".format(n = dataset_size, id = file_id, H = Hdiff))

def compare_trend_notrend_curr(max_delay = 300):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    for curr in ['CAD', 'CHF', 'GBP', 'JPY', 'USD']:
        df = parse_curr_csv(curr)
        ax1.scatter(df.date, df.value, s = 1, label=curr)

        delays = range(max_delay+1)
        acorr = sm.tsa.acf(df.value, nlags=max_delay)
        ax2.plot(delays, acorr, label=curr)

        notrend = remove_trends(df, False)
        ax3.scatter(notrend.date, notrend.value, s = 1, label=curr)

        acorr_notrend = sm.tsa.acf(notrend.value, nlags=max_delay)
        ax4.plot(delays, acorr_notrend, label=curr)
        
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()

def remove_jumps_xz(file_id = 0, dataset_size = 100000, threshold = 7500, do_plot = False, do_write = True):
    df = parse_xz(file_id, dataset_size, True)
    filtered = df[df.diffs < threshold].copy().reset_index(drop=True)
    filtered.nums = np.cumsum(filtered.diffs)
    
    if(do_plot):
        fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2)
        ax1.scatter(range(len(df.diffs)), df.diffs, s = 1)
        ax2.plot(range(len(df.diffs)), df.nums)
        ax3.scatter(range(len(filtered.diffs)), filtered.diffs, s = 1)
        ax4.plot(range(len(filtered.diffs)), filtered.nums)

    if(do_write):
        filtered.to_csv("xz{id}_nojumps.csv".format(id = file_id))

    return filtered

def parse_timestamps( dataset_size=100000, compute_diffs = True):
    df = pd.read_csv("timestamps.csv", nrows=dataset_size)
    df.rename(columns={"Date & time": "timestamp"}, inplace=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    if(compute_diffs):
        df["diffs"] = np.concatenate(( np.array([0]), np.diff(df.timestamp)))

    print(df)

    return df

def scale_independence_xz(file_id = 0, dataset_size = 100000, n_bins=100, factor = 3):
    df = remove_jumps_xz(file_id=file_id, dataset_size=dataset_size)
    time_unit = np.array(df.nums)[-1]/(n_bins*factor)

    big_bins = np.zeros(n_bins)
    small_bins = np.zeros(n_bins*factor)

    for i in range(n_bins):
        for j in range(factor):
            lower_limit = i*factor*time_unit + j*time_unit
            upper_limit = lower_limit+time_unit
            count = len(df[(df.nums >= lower_limit) & (df.nums < upper_limit)])
            small_bins[i*factor+j] = count
            big_bins[i] += count
    
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.bar(range(n_bins), big_bins, width = 1)
    ax2.bar(range(n_bins*factor), small_bins, width = 1)


scale_independence_xz()
plt.show()