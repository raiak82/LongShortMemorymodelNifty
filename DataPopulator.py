# -*- coding: utf-8 -*-
"""
DataPopulator.py

This is script to load the NSE data, apply technical indicators
Moreover, it also load the data of other indices using multiple predefined CSVs and create a single data frame
"""
import pandas as pd
import gc
from talib import RSI, BBANDS, MACD, EMA, STOCH, ADX
import matplotlib.pyplot as plt
from nsepy import get_history
from datetime import date
oldstartdate = date(2004,1,1)
startdate = date(2005,1,1)
enddate = date(2019,4,4)

df=pd.DataFrame(get_history(symbol="NIFTY",start = oldstartdate, end = enddate,index = True))


EMA_200 = EMA(df['Close'],200)
EMA_100 = EMA(df['Close'],100)
EMA_50 = EMA(df['Close'],50)
EMA_21 = EMA(df['Close'],21)
EMA_5 = EMA(df['Close'],5)
df['EMA-200']=EMA_200
df['EMA-100']=EMA_100
df['EMA-50']=EMA_50
df['EMA-21']=EMA_21
df['EMA-5']=EMA_5
df['STOCH_slowk'] ,df['STOCH_slowd'] = STOCH(df['High'], df['Low'], df['Close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
df['MACD'], df['macdEMA'], df['macdHistory'] = MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['RSI-14']=RSI(df['Close'],timeperiod=14)
df['BollingerUpperBand'], df['BollingerMiddleBand'], df['BollingerLowerBand'] = BBANDS(df['Close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
"""ADX Data seems to be incorrect"""
df['ADX']= ADX(df['High'],df['Low'],df['Close'],timeperiod=14)  
df.index = (pd.to_datetime(df.index))
df= df[startdate:]

""" Importing Brent Crude price Data from CSV"""
df1=pd.read_csv("BrentCrude.csv")


""" Convert column name Date from String datatype to Date datatype"""
df1['Date']=pd.to_datetime(df1['Date'])

""" Sort the Sheet based on Date ascending order. This is one of pre-requisite to merge different dataframe.
Dataframe that has to be merged should be sorted by join columns. In this case, join column is date.
Hence, it is sorted based on Date column """
df1=df1.sort_values(by=['Date'],ascending=True)
df=pd.merge_asof(df,df1,on='Date')

""" Importing Dow Historical price Data from CSV. And perform similar action like BrentCrude price data"""
df1=pd.read_csv("DowHistorical.csv")
df1['Date']=pd.to_datetime(df1['Date'])
df1=df1.sort_values(by=['Date'],ascending=True)
df=pd.merge_asof(df,df1,on='Date')


""" Importing FTSE Historical price Data from CSV. And perform similar action like BrentCrude price data"""
df1=pd.read_csv("FTSE_Historical.csv")
df1['Date']=pd.to_datetime(df1['Date'])
df1=df1.sort_values(by=['Date'],ascending=True)
df=pd.merge_asof(df,df1,on='Date')


""" Importing Hang Seng Historical price Data from CSV. And perform similar action like BrentCrude price data"""
df1=pd.read_csv("HangSeng_Historical.csv")
df1['Date']=pd.to_datetime(df1['Date'])
df1=df1.sort_values(by=['Date'],ascending=True)
df=pd.merge_asof(df,df1,on='Date')

""" Importing USD_INR Historical price Data from CSV. And perform similar action like BrentCrude price data"""
df1=pd.read_csv("USD_INR_Historical.csv")
df1['Date']=pd.to_datetime(df1['Date'])

df1=df1.sort_values(by=['Date'],ascending=True)
df=pd.merge_asof(df,df1,on='Date')

df2=pd.DataFrame(get_history(symbol="INDIAVIX",start = startdate, end = enddate,index = True))
df2 = df2.rename({'Close': 'IndiaVIX'}, axis='columns')
df2.drop(['Open', 'High','Low','Previous','Change','%Change'], axis=1, inplace=True)

df2.index = (pd.to_datetime(df2.index))
df2=df2.sort_values(by=['Date'],ascending=True)
df2.dropna(inplace=True)
df=pd.merge_asof(df,df2,on='Date')

plt.plot(df['Date'],df['Close'])
# plotting the points  
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.update

df2=pd.read_excel("FII-Index.xlsx")
df2['Date']=pd.to_datetime(df2['Date'])
#df2.dropna(inplace=True)
df2=df2.sort_values(by=['Date'],ascending=True)
df=pd.merge_asof(df,df2,on='Date')



df.to_csv('NSE-Data.csv')


# naming the x axis 
plt.xlabel('Date') 
# naming the y axis 
plt.ylabel('NIFTY') 
  
# giving a title to my graph 
plt.title('NIFTY HISTORICAL DATA') 
  
# function to show the plot 
plt.show() 



# =============================================================================
# 
# del df1, df2, enddate, oldstartdate, startdate
# gc.collect()
# df1=pd.DataFrame()
# =============================================================================
