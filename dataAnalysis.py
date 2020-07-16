#!/bin/env python3

# Copyright 2020, Micah Gale. See license.

from datetime import date
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

cases = pd.read_csv("Trendlinear_Full_Data_data.csv", encoding="UTF-8")

cases = cases.replace(np.NaN, 0)
cases["Date"] = matplotlib.dates.datestr2num(cases["Date"])
cases["totalOnset"] = cases["Onset Confirmed"] + cases["Onset Probable"]

phases = pd.read_csv("phases.csv", encoding="UTF-8")


def graphByOnset(cases, phases):
    N = 7  # rolling window
    rolling_mean = np.convolve(cases["totalOnset"], np.ones((N,)) / N, mode="same")
    plt.plot_date(
        cases["Date"], cases["totalOnset"], xdate=True, label="Raw Data",
    )
    plt.plot_date(
        cases["Date"], rolling_mean, xdate=True, fmt="-", label="7 day moving average",
    )
    maxVal = cases.loc[(cases["totalOnset"]).idxmax()]
    maxVal = maxVal["totalOnset"]
    plt.xlabel("Date")
    plt.ylabel("Number of cases contracted")
    graphPhases(phases, maxVal)
    plt.legend(loc="center left")
    plt.show()


def graphPhases(phases, maxVal):
    for index, row in phases.iterrows():
        start = matplotlib.dates.datestr2num(row["start"])
        try:
            end = matplotlib.dates.datestr2num(row["end"])
        except TypeError:
            end = matplotlib.dates.date2num(date.today())
        middle = (start + end) / 2
        plt.text(
            middle,
            maxVal + 15 + 5 * (index % 2),
            row["period"],
            horizontalalignment="center",
        )
        plt.vlines(start, 0, maxVal + 20)


def spliceCasesByPhases(cases, phases):
    outputSplice = {}
    for index, row in phases.iterrows():
        start = matplotlib.dates.datestr2num(row["start"])
        try:
            end = matplotlib.dates.datestr2num(row["end"])
        except TypeError:
            end = matplotlib.dates.date2num(date.today())
        inRange = cases[cases["Date"] >= start]
        inRange = inRange[inRange["Date"] < end]
        outputSplice[row["period"]] = inRange
    return outputSplice


def fitCasesDecay(splices):
    dataSet = splices["Idaho Stay Home"]
    dataSet["DateFromStart"] = dataSet["Date"] - dataSet.iloc[0]["Date"]
    xdata = dataSet["DateFromStart"]
    N=7
    rolling_mean = np.convolve(dataSet["totalOnset"], np.ones((N,)) / N, mode="same")
    ydata = dataSet["totalOnset"]
    plt.plot(xdata,ydata, label="Raw Data")
    popt, pcov = curve_fit(expFunc, xdata, ydata)
    residuals = ydata - expFunc(xdata, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.plot(
        xdata,
        expFunc(xdata, *popt),
        label="{:.2f}*exp(-{:.2f}*x)+{:.2f} R^2={:.2f}".format(*popt,r_squared),
    )
    plt.legend()
    plt.ylabel("Number of cases contracted")
    plt.xlabel("Days since start of State-wide stay home order")
    plt.show()


def expFunc(x, a, b, c):
    return a * np.exp(-b * x) + c


graphByOnset(cases, phases)
splices = spliceCasesByPhases(cases, phases)
#fitCasesDecay(splices)
