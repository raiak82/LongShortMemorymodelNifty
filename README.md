# Machine Learning- Predictive Model for Nifty using Time Series modelling LSTM
LSTM model for Nifty - trained and tested with last 15 years data (with all key technical indicators and other important variables)

Nifty Data Mining and Pre-processing-
1) NSE (National stock exchange) provides open library NSE.py to pull historical and live data of Nifty values such as Nifty Open, High, low , Close, Volume and Turnover data. It is used by almost every algo-trading platform to fetch the live/historical Nifty data 
Refer to the NSE documentation https://nsepy.readthedocs.io/en/latest/ on how to use the api
2) Upon fetching Nifty records, TALIB api is used to calculate technical indicator values such as Exponential Moving Average, MACD, RSI -14, Bollinger Bands values. This is again used by every brokerage firm (you name it) to plot technical charts of Nifty 
Refer to TALIB documentation https://mrjbq7.github.io/ta-lib/doc_index.html on how to use the api 
3) Extract historical values of other important variables such as  Brent Crude Price, USD-INR value, London Stock Exchange values (FTSE index), USA stock exchange price (Dow index), Hong Kong stcok exchange (Hang Seng Index) and India Volatility Index (VIX) from Investing.com (https://www.investing.com/)
4) Extract historical Foreign Institutional Investors (FII) data such as Foreign Institutional Investors (FII) future index values and Foreign Institutional Investors (FII) Option index values from https://www.way2wealth.com/derivatives/fiiactivity/
 Note- Historical records are extracted and saved using automated selenium test scripts, as data from the site (https://www.way2wealth.com/derivatives/fiiactivity/) fetched in tabular format with 20 records displayed each time
5) Finally, all the data mined from different sources are merged together to create a single dataframe.
Features:- Nifty, Volume, Turnover, EMA-200, EMA-100, EMA-50, EMA-21, EMA-5, MACD, RSI-14, BollingerUpperBand, BollingerMiddleBand, BollingerLowerBand, Brent Crude Price, Dow Price, FTSE, Hang Seng Price, USD-INR price, Volatility Index VIX, FII Index Future Net, FII Index Options



Predictive ML Modeling- (LSTM- Multi-variate, multi-lag time-steps model)
1) Feature selection 
Correlation of each independent variable is analyzed using multiple feature selection techniques SelectKBest, Logistic Regression and ExtraTreeClassification. 
Outcome of the technique filters out less significance variable from list of features
2) Apply time steps lag of 5 which includes the features of previous 5 time steps as input to model
3) Design and train the LSTM model on training set and test it on test set
4) Plot the Actual v/s Predicted Nifty Value
