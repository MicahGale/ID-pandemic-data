#!/bin/env python3

# Copyright 2020, Micah Gale. See license.

from datetime import date
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cases = pd.read_csv("Trendlinear_Full_Data_data.csv", encoding="UTF-8")

cases = cases.replace(np.NaN,0)

phases = pd.read_csv("phases.csv", encoding="UTF-8")


def graphByOnset(cases, phases):
    cases["totalOnset"] = cases["Onset Confirmed"] + cases["Onset Probable"]
    N = 7  # rolling window
    rolling_mean = np.convolve(cases["totalOnset"], np.ones((N,)) / N, mode="same")
    plt.plot_date(
        matplotlib.dates.datestr2num(cases["Date"]), cases["totalOnset"], xdate=True, label="Raw Data"
    )
    plt.plot_date(
        matplotlib.dates.datestr2num(cases["Date"]), rolling_mean, xdate=True, fmt="-", label="7 day moving average"
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


graphByOnset(cases, phases)
