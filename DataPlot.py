# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:21:00 2018

@author: Ashutosh
"""
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_yearwise(df, name):
    df1 = df.resample('AS').count() # AS-Year Start
    fig, ax = plt.subplots(figsize=(16, 8))
    plt.yticks(np.arange(0, df1['Title'].max() + 20, 20))
    plt.xticks(df1.index)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    #plt.grid(True)
    #ax.grid(color='black', zorder=0)
    ax.bar(df1.index, df1['Title'], width=100, zorder=3, color='#1b9e77')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of articles', fontsize=12)
    plt.title("Number of articles published on HIV by "+name+" year-wise")
    # plt.show()
    plt.savefig('Plot_Yearly'+name+'.png')

def plot_monthwise(df, name):
    df_m = df.resample('M').count()
    df_quat = df.resample('Q').count() # Resample data to Semi-Annual
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.yticks(np.arange(df_m['Title'].min(), df_m['Title'].max() + 1, 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    ax.bar(df_m.index, df_m['Title'], width=20, zorder=10, color='#7570b3')
    plt.xticks(df_quat.index, rotation=70)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of articles', fontsize=12)
    plt.title("Number of articles published on HIV by " + name + " month-wise")
    # plt.show()
    plt.savefig('Plot_Monthly'+name+'.png')


# Functions to clean dates
def TOI(df):
    df.dropna(how='any', axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Cleaning Dates
    df['Date'] = df['Date'].apply(lambda x: re.findall(r'[a-zA-Z]{3}\s[0-9]{1,2},\s[0-9]{4}', x))
    df['Date'] = df.Date.apply(', '.join)
    df['Date'] = df['Date'].str.replace(',', '').astype('str')
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%b %d %Y'))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.reset_index(drop=True).set_index('Date')
    df.index = pd.to_datetime(df.index)
    return df

if __name__ == "__main__":
    df = pd.read_csv('HIV.csv')
    df = TOI(df)
    plot_yearwise(df, 'TOI')
    plot_monthwise(df, 'TOI')
    print(df.head())