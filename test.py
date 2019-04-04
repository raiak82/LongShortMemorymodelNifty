# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 20:44:44 2019

@author: raiak
"""
import pandas as pd
import numpy as np


# Importing the dataset
dataset = pd.read_csv('NSE-Data.csv')

X = dataset.iloc[:, 1:22]
y = dataset.iloc[:, -1]


from sklearn.preprocessing import MinMaxScaler

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(X)

x_train, y_train = [], []
for i in range(60,len(X)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))



print (X_train.shape[1],1)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import  Dense, Dropout, LSTM, CuDNNLSTM


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(LSTM(100, input_shape=(x_train.shape[1],1), activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=True)) # 100 num of LSTM units

classifier.add(LSTM(100, activation='tanh', recurrent_activation='hard_sigmoid'))

classifier.add(CuDNNLSTM(units=128))
classifier.add(Dropout(0.1))
classifier.add(Dense(output_dim=1, activation = 'sigmoid'))
print(classifier.summary())



# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

X_train= (100,X_train)

# Fitting the ANN to the Training set
classifier.fit(x_train, y_train,  batch_size=1000, epochs = 10, verbose=2)


