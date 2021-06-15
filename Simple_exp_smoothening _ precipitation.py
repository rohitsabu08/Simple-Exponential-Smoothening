# Importing the libraries required for Simple Exp.Smoothening
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error  
from math import sqrt 

# read in the dataset, split as train and test data(ratio = 80:20)
series = read_csv('precipitation.csv') #Yearly records of precipitation dataset
train = series.iloc[0:80]   #Splitting the test and train dataset
test = series.iloc[80:]

####  MODEL FITTING  ####
# fit a SES with a smoothening level of 0.1 and reducing the noise u=in the dataset
fitx = SimpleExpSmoothing(np.asarray(train['rain'])).fit(smoothing_level=0.1,
                                                         optimized=False)
# Calculating the RMSE of train data fit on the model:
rms_train = sqrt(mean_squared_error(train['rain'], fitx.fittedvalues)) 
print('rmse for fitted values with optimal alpha: ', rms_train)

# Plot training data against fitted values using both models
plt.figure(figsize=(8,3))
plt.plot(fitx.fittedvalues, label = 'fitted opt')
plt.plot(train['rain'], label='train')
plt.legend(loc='best')
plt.show()

####   FORECAST   ####
# forecast using the SES model:
test['SES_opt_fcast'] = fitx.forecast(len(test))
# plot the time series as train, test and forecast series
plt.figure(figsize=(8,3))
plt.plot(train['rain'], label='train data')
plt.plot(test['rain'], label='test data')
plt.plot(test['SES_opt_fcast'], label='SES forecast')
plt.legend(loc='best')
plt.show()
# calculate RMSE of the forecast on test data
rms_opt = sqrt(mean_squared_error(test.rain, test.SES_opt_fcast)) 
print(test)
print('RMSE for model: ', rms_opt)

