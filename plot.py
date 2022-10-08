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


def parse_dict_year():  
    deviza = {}
    with open('data-deviza.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        spamreader.__next__()
        for row in spamreader:
            row = row[0].split(',')

            date = dt.datetime.strptime(row[0].strip('"'),'%Y-%m-%d').date()
            year = date.year
            currency = row[1].strip('"')
            val = (float)(row[2].strip('"'))

            date = int(date.strftime("%j"))

            if currency in deviza:
                if year in deviza[currency]:
                    deviza[currency][year][date] = val
                else:
                    deviza[currency][year] = {date:val}
            else:
                deviza[currency] = {year:{date: val}}

        return deviza

def parse_dict():  
    deviza = {}
    with open('data-deviza.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        spamreader.__next__()
        for row in spamreader:
            row = row[0].split(',')

            date = dt.datetime.strptime(row[0].strip('"'),'%Y-%m-%d').date()
            currency = row[1].strip('"')
            val = (float)(row[2].strip('"'))

            if currency in deviza:
                deviza[currency][date] = val
            else:
                deviza[currency] = {date: val}
            

    return deviza

def graph_random_year():
    deviza = parse_dict_year()

    usd = deviza["USD"]

    for year in usd.keys():
        x = np.array(list(usd[year])).reshape((-1,1))
        y = np.array(list(usd[year].values()))

        model = LinearRegression().fit(x,y)

        usd[year]["model"]=model

    year1 =random.randint(2000,2020)

    data1 = usd[year1].copy()

    model1 = data1.pop("model", None)

    days1 = data1.keys()

    intercept1 = model1.intercept_
    slope1 = model1.coef_
    avg1 = np.average(list(data1.values()))

    data1_no_trend = [data1[key]-intercept1-key*slope1+avg1 for key in list(data1.keys())]

    plt.title(year1)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.yticks(np.arange(0, 310, 20))
    plt.scatter(days1,data1.values(), s=1, color='red')
    plt.scatter(data1.keys(),data1_no_trend, s=1, color='black')
    plt.gcf().autofmt_xdate()
    plt.axline((0,intercept1), slope = slope1, color='red')

    plt.show()

def remove_trends(plot):
    df = parse_csv()
    ts = df['value']

    breaks = pelt(ts)

    breaks_rpt = []
    notrend = []

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

        notrend += list(y-p)

        if(plot):
            plt.plot(d,p,c='grey')
            plt.scatter(d,y,s=1)
            breaks_rpt.append(ts.index[h-1])

    if(plot):
        breaks_rpt = pd.to_datetime(breaks_rpt)

        for i in breaks_rpt:
            plt.axvline(i, color='red',linestyle='dashed')
        plt.grid()
        plt.legend()

        plt.subplot(1,2,2)
        plt.scatter(df.date.values, notrend, s=1)
        plt.show()

    return notrend

def parse_csv():
    df = pd.read_csv('data-deviza.csv')
    currency = input("Which currency would you like to see?")
    df['date'] = pd.to_datetime(df['date'])
    df = df[df["iso_4217"] == currency ]
    df = df[df['date'] > dt.datetime(2000,1,1)]
    df.set_index(df['date'], inplace = True)
    return df

def pelt(ts):
    plt.subplot(1,2,1)

    y0 = np.array(ts.tolist())

    breakpoint_model = rpt.Pelt(min_size=30,model="rbf")
    breakpoint_model.fit(y0)
    return [0] + breakpoint_model.predict(pen=10)


def jenks(ts, n_breaks, y):
    breaks = jenkspy.jenks_breaks(y, n_classes=n_breaks)
    breaks_jkp = []
    for v in breaks:
        idx = ts.index[ts == v][0]
        breaks_jkp.append(idx)
    

    plt.scatter(ts.index, ts, s=1, label='data')
    plt.title('USD')
    print_legend = True
    for i in breaks_jkp:
        if print_legend:
            plt.axvline(i, color='red',linestyle='dashed', label='breaks')
            print_legend = False
        else:
            plt.axvline(i, color='red',linestyle='dashed')
    plt.grid()
    plt.legend()
    plt.show()

notrend = remove_trends(True)
