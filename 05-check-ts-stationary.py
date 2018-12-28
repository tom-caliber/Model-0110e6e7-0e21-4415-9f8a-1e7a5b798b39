'''
Notes: 
More negative the test statistic value more likely 
we are to reject the null hypothesis => time series is stationary.
'''
from sygmoid.imports import *
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from numpy import log

PATH_TO_DATA = '../datasets/04-ts.csv'

def check_ts_stationary(PATH_TO_DATA):
	series = pd.read_csv(PATH_TO_DATA, header=0, parse_dates=[0], index_col=0, squeeze=True)
	Vals = series.values

	# Augmented Dicky-Fuller test on data for testing stationarity
	result = adfuller(Vals)
	print('\nADF Statistic (more negative implies stationarity): %f' % result[0])
	print('p-value: %f' % result[1])

	# Augmented Dicky-Fuller test on log transformed data
	logV = log(Vals)
	result = adfuller(logV)
	print('\nADF Statistic of log-transformed data  (more negative implies stationarity): %f' % result[0])
	print('p-value: %f' % result[1])

	print('\nCritical Values (less than x% implies lower probability of statistical test being fluke):')
	for key, value in result[4].items():
		print('\t%s: %.3f' % (key, value))

check_ts_stationary(PATH_TO_DATA)