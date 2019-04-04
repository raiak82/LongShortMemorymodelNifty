from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

 
# load dataset
dataset = read_csv('NSE-Data2.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')


# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
values=scaled
 
# split into train and test sets
n_train_hours = 3000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

 
# design network
model = Sequential()
model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2]),bias_initializer="zeros"))
model.add(Dense(1))

# fit network
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])

 # fit network
history = model.fit(train_X, train_y, epochs=100, batch_size=30, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((test_X[:, :], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
next_nifty=inv_yhat[-1]
prev_nifty=inv_yhat[-2]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:, :], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
inv_y = inv_y[:-1]


pyplot.figure(figsize=(18, 18))
pyplot.plot(inv_y, label='Nifty')
pyplot.plot(inv_yhat, label='Nifty predicted')
pyplot.legend()
pyplot.show()   

print("The next predicted NIFTY trend is from ", prev_nifty, " to ", next_nifty)                                   