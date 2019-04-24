
#import libraries
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from bs4 import BeautifulSoup
from textblob import TextBlob
import string
from numpy import concatenate



from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import concat


def series_to_new_supervised(df, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(df) is list else df.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    #names = df.columns.values
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(' %s(t-%d)' % (df.columns[j], i)) for j in range(n_vars)]

    	# forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(' %s(t)' % (df.columns[j])) for j in range(n_vars)]
        else:
            names += [(' %s(t+%d)' % (df.columns[j], i))for j in range(n_vars)]
    	# put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    	# drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def analyze_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(tweet)
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1
    

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    try:
        tweet = BeautifulSoup(tweet, 'lxml')
    except:
        tweet = tweet
    tweet = tweet.str.replace(r'[^\x00-\x7F]+', '')
    tweet = tweet.str.replace(r'(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(https?://[A-Za-z0-9./]+)', " ")
    table = str.maketrans(dict.fromkeys(string.punctuation))
    tweet = tweet.str.translate(table)
    try:
        tweet = tweet.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        tweet = tweet
    return tweet



def remove_columns_from_transformed_series(df2, main_column):
    columns_to_be_removed=[]
    for i in range(0,df2.shape[1]):
        string1=df2.columns[i]
        if re.search("\(t\-",string1): 
            if not (re.search(main_column,string1)):
                columns_to_be_removed.append(string1)
        if re.search("\(t\+",string1):
            if not (re.search(main_column,string1)):
                columns_to_be_removed.append(string1)
    df2.drop(columns_to_be_removed, axis=1, inplace=True)
    return df2


#load the dataset
df_train = pd.read_csv("train_timeseries2.csv", index_col = 'date')
df_train.head()

df_train_tweets=pd.read_csv('train_tweets.csv',encoding="ISO-8859-1",parse_dates=['created_date'])

df_train_tweets['text']=clean_tweet(df_train_tweets['text'])
df_train_tweets['SA'] = np.array([ analyze_sentiment(tweet) for tweet in df_train_tweets['text'].astype(str) ]) 
columns_from_tweets_to_be_removed=['tweet_id','text','retweet_count','favorite_count','follower_count','account']
df_train_tweets.drop(columns_from_tweets_to_be_removed, axis=1, inplace=True)
df_train_tweets['created_date'] = pd.to_datetime(df_train_tweets['created_date'], errors='coerce')
df_train_tweets['created_date'] = pd.to_datetime(df_train_tweets['created_date']).dt.date
df_train_tweets=df_train_tweets.groupby(['created_date','SA']).size().reset_index().groupby(['created_date','SA'])[[0]].max().unstack()

#changing from multi index dataset to single index dataset
mi= df_train_tweets.columns
ind = pd.Index([e[0] + e[1] for e in mi.tolist()])
df_train_tweets.columns = ind
df_train_tweets=df_train_tweets.fillna(0)
#rename columns of dataset from -1,0,1 to Negative, neutral and positive
df_train_tweets.rename(columns={-1:'Negative_tweets',0:'Neutral_tweets',1:'Positive_tweets'}, inplace= True)
df_train_tweets.index.name = 'date'


df_train.index = pd.to_datetime(df_train.index)
df_train.sort_index(inplace=True, ascending=True)

df_train_tweets.index = (pd.to_datetime(df_train_tweets.index))
df_train_tweets.sort_index(inplace=True, ascending=True)

df_train=pd.merge_asof(df_train,df_train_tweets,on='date')
df_train = df_train.set_index('date')


df_test = pd.read_csv("test_timeseries2.csv")
df_test.head()


df_test_tweets=pd.read_csv('test_tweets.csv',encoding="ISO-8859-1",parse_dates=['created_date'])
df_test_tweets['text']=clean_tweet(df_test_tweets['text'])
df_test_tweets['SA'] = np.array([ analyze_sentiment(tweet) for tweet in df_test_tweets['text'].astype(str) ]) 
df_test_tweets['created_date'] = pd.to_datetime(df_test_tweets['created_date'], errors='coerce')
df_test_tweets['created_date'] = pd.to_datetime(df_test_tweets['created_date']).dt.date
df_test_tweets.drop(columns_from_tweets_to_be_removed, axis=1, inplace=True)
df_test_tweets=df_test_tweets.groupby(['created_date','SA']).size().reset_index().groupby(['created_date','SA'])[[0]].max().unstack()

#changing from multi index dataset to single index dataset
mi= df_test_tweets.columns
ind = pd.Index([e[0] + e[1] for e in mi.tolist()])
df_test_tweets.columns = ind
df_test_tweets=df_test_tweets.fillna(0)
#rename columns of dataset from -1,0,1 to Negative, neutral and positive
df_test_tweets.rename(columns={-1:'Negative_tweets',0:'Neutral_tweets',1:'Positive_tweets'}, inplace= True)

df_test = df_test.set_index('date')
df_test.index = pd.to_datetime(df_test.index)
df_test.sort_index(inplace=True, ascending=True)


df_test_tweets.index.name = 'date'
df_test_tweets.index = (pd.to_datetime(df_test_tweets.index))
df_test_tweets.sort_index(inplace=True, ascending=True)

df_test=pd.merge_asof(df_test,df_test_tweets,on='date')
df_test = df_test.set_index('date')


del df_train_tweets, columns_from_tweets_to_be_removed, df_test_tweets 


##############################################

look_back = 5
look_forward = 2
df_train = series_to_new_supervised(df_train, look_back, look_forward)
df_train=remove_columns_from_transformed_series(df_train,'close')

##################

#df_test = series_to_new_supervised(df_test, look_back, 2)
#df_test=remove_columns_from_transformed_series(df_test,'open')

####################

df_train.to_csv('df_train.csv')
df_test.to_csv('df_test.csv')


dataset_train = df_train.values
dataset_train = dataset_train.astype('float32')

scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train = scaler.fit_transform(dataset_train)

#=================================================================

#dataset_test = df_test.values
#dataset_test = dataset_test.astype('float32')
#
#scaler = MinMaxScaler(feature_range=(0, 1))
#dataset_test = scaler.fit_transform(dataset_test)

#=================================================================

n_train_hours = 700
train = dataset_train[:n_train_hours, :]
test = dataset_train[n_train_hours:, :]

# split into input and outputs
trainX, trainY = train[:, :-1], train[:, -1]
testX, testY = test[:, :-1], test[:, -1]



# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#Building Model
model = Sequential()
model.add(LSTM(500, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=256, verbose=2, validation_data=(testX, testY))
model.summary()

#trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


testX = testX.reshape((testX.shape[0], testX.shape[2]))
invTestY = concatenate((testX[:, :], testPredict), axis=1)
invTestY = scaler.inverse_transform(invTestY)
predictedtestY=invTestY[:,-1]

# invert scaling for actual
testY = testY.reshape((len(testY), 1))
invActualY = concatenate((testX[:, :], testY), axis=1)
invActualY = scaler.inverse_transform(invActualY)
ActualY = invActualY[:,-1]

testScore = math.sqrt(mean_squared_error(ActualY, predictedtestY))
print('Test Score: %.2f RMSE' % (testScore))


pyplot.figure(figsize=(18, 18))
pyplot.plot(ActualY, label='Nifty')
pyplot.plot(predictedtestY, label='Nifty predicted')
pyplot.legend()
pyplot.show()   






