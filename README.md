Trend Prediction of NIFTY-50

Purpose- the Intention of creating the Machine Learning model of NIFTY-50 is to predict the next day trend of NIFTY-50 by training/testing the model with 15 years past historical records of NIFTY-50 along with other key dependent variables (Brent crude price, Dollar rate, Hang Seng Index, Dow Index, FTSE index and FII daily activity)
GitHub project- https://github.com/raiak82/LongShortMemorymodelNifty
The model creation and testing is 2 stage process-
Step-1: Nifty Data Mining and Pre-processing steps: 
•	NSE (National stock exchange) provides open library NSE.py to pull historical and live data of Nifty values such as Nifty Open, High, Low, Close, Volume and Turnover data. It is used by almost every Algo-trading platform to fetch the live/historical Nifty data 

Refer to the NSE documentation https://nsepy.readthedocs.io/en/latest on how to use the API
 
•	Upon fetching Nifty records, TALIB api is used to calculate technical indicator values such as Exponential Moving Average, MACD, RSI -14, Bollinger Bands values. This is again used by every brokerage firm (you name it) to plot intuitive technical charts of Nifty index
 
Refer to TALIB documentation https://mrjbq7.github.io/ta-lib/doc_index.htmlon how to use the API 
 
•	Extract historical values of other important variables such as  Brent Crude Price, USD-INR value, London Stock Exchange values (FTSE index), USA stock exchange price (Dow index), Hong Kong stock exchange (Hang Seng Index) and India Volatility Index (VIX) from Investing.com (https://www.investing.com/ )
 
•	Extract historical Foreign Institutional Investors (FII) data such as Foreign Institutional Investors (FII) future index values and Foreign Institutional Investors (FII) Option index values from  https://www.way2wealth.com/derivatives/fiiactivity/ 
 
Note- Historical records are extracted and saved using automated selenium test scripts, as data from the site (https://www.way2wealth.com/derivatives/fiiactivity/ ) fetched in tabular format with 15 records displayed each time which takes ~8-10 hours to fetch 15 years record of FII data. This is one time exercise to pull the historical records of FII data.


5) Finally, all the data mined from different sources are merged together to create a single data frame.


Features
•	Nifty-50
•	Volume
•	Turnover
•	EMA-200
•	EMA-100
•	EMA-50
•	EMA-21
•	EMA-5
•	MACD
•	RSI-14
•	BollingerUpperBand
•	BollingerMiddleBand
•	BollingerLowerBand
•	Brent Crude Price
•	Dow Price
•	FTSE
•	Hang Seng Price
•	USD-INR price
•	Volatility Index VIX
•	Foreign Institutional Investors (FII) Index Future Net
•	Foreign Institutional Investors (FII) Index Options Net

Note: One of the important feature to predict the next day Nifty trend is Twitter Sentiment of hashtag #NIFTY
As the twitter API only provides feed of last 6 months, it is a challenge to include the sentiment of hashtag #Nifty from twitter feed to the model
Step-2:  Predictive ML Modelling- (LSTM- Multi-variate, multi-lag time-steps model)
1)	Once the data mined from all the sources, next important step is Feature selection which is done by correlation of each independent variable with dependant variable of next day Nifty.
Using multiple feature selection techniques Heatmap, SelectKBest, Logistic Regression and ExtraTreeClassification top 15 features are selected which shows maximum variance with next day Nifty-50 value

 

2)	After a feature reduction, time step delay of 5 time steps is applied to LSTM model with 500 nodes, epoch size 150 and batch size 10
Train on 3000 samples, validate on 569 samples
Epoch 1/150
 - 27s - loss: 0.0013 - mean_squared_error: 0.0013 - val_loss: 0.0180 - val_mean_squared_error: 0.0180
Epoch 2/150
 - 24s - loss: 0.0016 - mean_squared_error: 0.0016 - val_loss: 0.0043 - val_mean_squared_error: 0.0043
Epoch 3/150

4) Plot the Actual v/s Predicted Nifty Value

 

