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
from keras.layers import Masking
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('NSE-Data1.csv', header=0, index_col=0)
values = dataset.values
# ensure all data is float
values = values.astype('float32')


# frame as supervised learning
#reframed = series_to_supervised(values, 2, 1)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#reframed.drop(reframed.columns[[55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71]], axis=1, inplace=True)
# drop columns we don't want to predict
#reframed.drop(reframed.columns[[14,15,16,17,18,19,20,21,22,23,24,25]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]], axis=1, inplace=True)
#reframed.drop(reframed.columns[[37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]], axis=1, inplace=True)
values=scaled

#remove next day data to be predicted
next_day= values[-1,:]
 
# split into train and test sets
n_train_hours = 3000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]


# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

VAR_x=test_X
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
#train_y=train_y.reshape((train_y.shape[0], 1,1))
#test_y=test_y.reshape((test_y.shape[0], 1,1))

hidden_units=50
Dropout=0.1
 
 
# design network
model = Sequential()
model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2]),bias_initializer="zeros"))
#model.add(LSTM(200, activation='tanh', recurrent_activation='sigmoid'))
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

