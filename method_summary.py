# Plot the data, identify unusual observations & understand patterns (remove outliers, see trend/ seasonality, etc.)
# If necessary, use a Box-Cox transformation to stablize the variance (need investigation)
# Select the model
# Step 1: Test Stationarity (through charting data with time, 1st, 2nd order difference) = test whether mean and variation change with time
        # Unit Root Test
    # Strict Stationarity: 
    # Weak Stationarity (most common): 
        # 1st order diference: y(n) - y(n-1)
# If necessary, difference the data until it appears stationary. Use unit-root tests if unsure about stationarity (adf test)
# Step 2: Determine p 
        # Autocorrelation Function (ACF): ACF(k) = p(k) = Cov(y(t), y(t-k))/ Var(y(t)) in [-1, +1] (include 95% confidence interval)
        # Partial Autocorrelation Function (PACF): 
# Plot the ACF/PACF of the differenced data and try to determine the possible candicate models (p, d, q)
# Step 3: AutocorrelAtion exist
#
# Try the chosen models (p, d, q) and use the AIC to search for a better model
#
# Check the residuals from the chosen model by plotting the ACF of the residuals, and doing a port-manutrau test of the residuals (need)
# Check whether the residual look like white noise
        #
""" AR(p) (Autoregressive model): y(t) = a0 + a1*y(t-1) + a2*y(t-2) + a3*y(t-3) + ... + ap*y(t-p) + error(t)
        # 
    MA(q) (Moving-average model): y(t) = a0 + error(t) + b1*error(t-1) + b2*error(t-2) + b3*error(t-3) + ... + bq*error(t-q)
            moving average method can effectively eliminate the influences from random fluctuations
            y(t) depends only on the lagged forecast errors
        # 1. 
        # 2. 
    ARMA Model: y(t) - [a1*y(t-1) + a2*y(t-2) + a3*y(t-3) + ... + ap*y(t-p)] = 
                                    [error(t) + b1*error(t-1) + b2*error(t-2) + b3*error(t-3) + ... + bq*error(t-q)]
    ARIMA Model:
        # 1.
        # 2. Moving Average: the regression error is actually a linear combination of past error terms 
        # 3. Integrated: the data values have been replaced with the difference between their values and the previous values
            # this means how many times that we have to difference the data to get it stationary so that AT and MA components could work
    Linear Model: y(t) = a0 + a1x1(t) + a2x2(t) + ... anxn(t) + error(t)
"""
#
#
""" LSTM Model:
    # 1. 
    # 2. 
"""
#
""" ARIMA vs LSTM Model: (mechanism and accuracy comparison)
    # 1. 
    # 2. 
"""
#
# R-squared: coefficient of determination (in econometrics, this can be interpreted as the percentage of variance explained by the model)
# Mean Absolute Error: this is an interpretable metric because it has the same unit of measurment as the initial series
# Median Absolute Error: again, an interpretable metric that is particularly interesting because it is robust to outliers
# Mean Squared Error: the most commonly used metric that gives a higher penalty to large errors and vice versa
# Mean Squared Logarithmic Error: practically, this is the same as MSE, but we take the logarithm of the series. 
    # As a result, we give more weight to small mistakes as well. This is usually used when the data has exponential trends
# Mean Absolute Percentage Error: this is the same as MAE but is computed as a percentage, which is very 
    # convenient when you want to explain the quality of the model to management
# In[]:
""" ARIMA Model (p, d, q):
    # 1. Based on PACF, p can be determined, as the PACF values all fall into the Confidence Interval
    # 2. Based on ACF, q can be determined.
    # 3. Based on the order of difference, d can be decided
"""
#
""" AIC (Akaike Information Criterion): AIC = 2*l - 2*ln(L)
        # to punish over-complicated models
    BIC (Bayesian Information Criterion): 
"""
#
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(12,6)}, style = 'whitegrid', palette = 'Set1')
AXJO = yf.download('^AXJO', '2016-01-01', '2020-02-29')
AXJO['diff_1'] = AXJO['Close'].diff(1)
AXJO['diff_2'] = AXJO['diff_1'].diff(1)
#
plt.scatter(x = AXJO.index, y = AXJO['Close'])
plt.plot(AXJO['Close'])
plt.plot(AXJO['diff_1'])
plt.plot(AXJO['diff_2'])
#
import statsmodels.api as sm
#import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
#
fig = plt.figure(figsize = (10,6))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(AXJO['Close'], lags = 100, ax = ax1)
ax1.xaxis.set_ticks_position('bottom')
fig.tight_layout()
#
ax2 = fig.add_subplot(212) # 
fig = sm.graphics.tsa.plot_pacf(AXJO['Close'], lags = 100, ax = ax2) # PACF
ax2.xaxis.set_ticks_position('bottom')
fig.tight_layout()
#
def tsplot(y, lags = None, title = '', figsize = (10,6)):
    plt.figure(figsize = figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0))
    hist_ax = plt.subplot2grid(layout, (0, 1))
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    #
    y.plot(ax = ts_ax)
    ts_ax.set_title(title)
    y.plot(ax = hist_ax, kind = 'hist', bins = 50)
    hist_ax.set_title('Histogram')
    smt.graphics.plot_acf(y, lags = lags, ax = acf_ax)
    smt.graphics.plot_pacf(y, lags = lags, ax = pacf_ax)
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax
tsplot(AXJO['Close'], lags = 100, title = 'ASX 200 Share Index')
#
ts_train =AXJO['Close'][:-100]
arima200 = sm.tsa.SARIMAX(ts_train, order = (80,0,3))
model_results = arima200.fit()
#
import itertools
p_min = 0
d_min = 0
q_min = 0
p_max = 4
d_max = 0
q_max = 4
#
results_bic = pd.DataFrame(index = ['AR{}'.format(i) for i in range(p_min, p_max+1)],
    columns = ['MA{}'.format(i) for i in range(q_min, q_min + 1)])
for p, d, q in itertools.product(range(p_min, p_max + 1),
                                  range(d_min, d_max + 1),
                                  range(q_min, q_max + 1)):
    if p ==0 and q == 0 and d == 0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    try:
        model = sm.tsa.SARIMAX(ts_train, order = (p, d, q))
#                               enforce_stationarity = False, 
#                               enforce_invertibility = False
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
results_bic = results_bic[results_bic.columns].astype(float)
results_bic
#
fig, ax = plt.subplots(figsize = (10, 8))
ax = sns.heatmap(results_bic, mask = results_bic.isnull(), ax = ax, annot = True, fmt = '.2f', cmap = 'coolwarm')
ax.set_title('BIC')
#
train_results = sm.tsa.arma_order_select_ic(ts_train, ic = ['aic', 'bic'], trend = 'nc', max_ar = 4, max_ma = 4)
print('AIC ', train_results.aic_min_order)
print('BIC ', train_results.bic_min_order)
#
model_results.plot_diagnostics(figsize = (16, 12))


#
""" AR(1) Model: y(t) = a0 + a1*y(t-1) + error(t)
    If we assume that this model has a mean: miu(t) = a0 + a1*y(t-1), then we will have
    # 1. Conditional MeanL E(y|y(t-1)) = a0 + a1*y(t-1) = miu
    # 2. Var(y(t)|y(t-1)) = Var(a0 + a1*y(t-1) + error(t)| y(t-1)) = Var(error(t)|y(t-1)) = sigma**2
    # 
"""
#
# In[]:
#
import os
os.chdir('C:\Python_Project\Deep_Learning\Data')
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('norway_new_car_sales_by_model.csv', encoding = 'cp1252')
df['Date'] = pd.to_datetime(df.assign(Day=1).loc[:, ['Year','Month','Day']])
df = df[['Date','Quantity']]
df = df.groupby(['Date']).sum()
#
df['Quantity'].plot(color='red')
plt.xlabel('Date')
plt.ylabel('Quantity')
plt.title('Sales of cars')
#
from pandas.plotting import lag_plot # to observe the relationship between f(n) and f(n-1)
lag_plot(df)
plt.title('Plot of lag values 1')
#
from statsmodels.graphics.tsaplots import plot_acf # for plotting ACF
plot_acf(df, lags=36)
#
from statsmodels.graphics.tsaplots import plot_pacf # for plotting PACF
plot_pacf(df, lags=50)
#
# In[]:
# Functions that will be used during the model build phase
# This segment does not produce any output

from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas import Series
from pandas import concat

# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(len(n_in), 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, len(n_out)):
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

#data = df
#series_to_supervised(df, n_in=3, n_out=5, dropnan=True)

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
    # extract raw values
    raw_values = series.values
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    # rescale values to -1, 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(diff_values)
    scaled_values = scaled_values.reshape(len(scaled_values), 1)
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    # split into train and test sets
    train, test = supervised_values[0:-len(n_test)], supervised_values[-len(n_test):]
    return scaler, train, test

# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
    # reshape training into [samples, timesteps, features]
    X, y = train[:, 0:n_lag], train[:, n_lag:]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=n_batch, verbose=0, shuffle=False)
        model.reset_states()
    return model

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
    # reshape input pattern to [samples, timesteps, features]
    X = X.reshape(1, 1, len(X))
    # make forecast
    forecast = model.predict(X, batch_size=n_batch)
    # convert to array
    return forecast
 
# make all forecasts & append them
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
    forecasts = list()
    for i in range(len(test)):
        X, y = test[i, 0:n_lag], test[i, n_lag:]
        # make forecast
        forecast = forecast_lstm(model, X, n_batch)
        # store the forecast
        forecasts.append(forecast)
    return forecasts
 
    
# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # create array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, n_seq)
        # invert scaling
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        # store
        inverted.append(inv_diff)
    return inverted
 
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
    for i in range(n_seq):
        actual = [row[i] for row in test]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(actual, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))

       
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    plt.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s, off_e)]
        yaxis = [series.values[off_s]] + forecasts[i]
        plt.plot(xaxis, yaxis, color='red')
    # show the plot
#
# Automated tuning of hyperparameters using Grid Search #
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

# fix random seed for reproducibility
seed = 5
np.random.seed(seed)

# fit an LSTM network to training data
def fit_lstm_tuning(n_test = 28, n_lag = 1, n_batch = 1, n_neurons = 1, activation = 'sigmoid', optimizer = 'adam'):
    # design network
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(n_batch, 1, n_lag), activation = activation, stateful=True))
    model.add(Dense(y.shape[1]))
    #model.add(Activation('sigmoid'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model

# define the grid search parameters
n_lag = [1]
#n_lag = [*map(int, n_lag)]
n_seq = [1]
#n_epochs = [1500, 2000, 2500, 3000]
n_batch = [1]
n_neurons = [1,2,3]
activation = ['softmax', 'relu', 'tanh', 'sigmoid']
optimizer = ['SGD', 'RMSprop','Adam']
# configure
series = df
n_test = [26]

series = series.values.tolist()
scaler, train, test = prepare_data(series, n_test, n_lag, n_seq)
X_input, y_dependent = train[:, 0:len(n_lag)], train[:, len(n_lag):]
X, y = X_input, y_dependent
X = X.reshape(X.shape[0], 1, X.shape[1])

# create model
model = KerasClassifier(build_fn=fit_lstm_tuning, epochs = 1500, batch_size = 1, verbose=0)

param_grid = dict(n_batch = n_batch, n_neurons = n_neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# In[]: ARIMA model for time series forecasting
import warnings
import itertools
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
df.plot(figsize=(10, 6))
#time-series seasonal decomposition
from pylab import rcParams
rcParams['figure.figsize'] = 10, 6
#
decomposition = sm.tsa.seasonal_decompose(df, model='additive') # decompose time-series data into: seasonal + trend + residual
fig = decomposition.plot()
# Parameter Selection for the ARIMA Time Series Model
# Define the p, d and q parameters to take any value between 0 and 2
p = range(0, 3)
d = range(0, 3)
q = range(0, 1)
#Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
#Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
warnings.filterwarnings("ignore") # specify to ignore warning messages
# All combinations of parameters are used & best set of parameters with Minimum AIC is selected
AIC_list = pd.DataFrame({}, columns=['param','param_seasonal','AIC'])
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
            temp = pd.DataFrame([[ param ,  param_seasonal , results.aic ]], columns=['param','param_seasonal','AIC'])
            AIC_list = AIC_list.append( temp, ignore_index=True)  # 
            del temp
        except:
            continue
#
m = np.nanmin(AIC_list['AIC'].values) # Find minimum value in AIC
l = AIC_list['AIC'].tolist().index(m) # Find index number for lowest AIC
Min_AIC_list = AIC_list.iloc[l,:]

# Fitting ARIMA model wtih best parameters obtained using Minimum AIC
mod = sm.tsa.statespace.SARIMAX(df,
                                order=Min_AIC_list['param'],
                                seasonal_order=Min_AIC_list['param_seasonal'],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()

print("### Min_AIC_list ### \n{}".format(Min_AIC_list))
print(results.summary().tables[1])
#
results.plot_diagnostics(figsize=(12, 6))
#
# Validating Forecasts
pred_dynamic = results.get_prediction(start=pd.to_datetime('2016-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
#
ax = df['2007':].plot(label='observed') # 
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
#
ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2016-01-01'), df.index[-1],
                 alpha=.1, zorder=-1)
ax.set_xlabel('Date')
ax.set_ylabel('Car Sales')
plt.legend()
#Extract the predicted and true values of our time series
y_forecasted = pred_dynamic.predicted_mean
y_truth = df['2016-01-01':]

#Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
# Producing and Visualizing Forecasts
#Get forecast 'x' steps ahead in future
pred_uc = results.get_forecast(steps=24)
#Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
#plot the time series and forecasts of its future values
ax = df.plot(label='observed', figsize=(10, 6))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Car Sales')
plt.legend()
#
# In[]:
#
""" LSTM Model: 
    # 1.
    # 2.
"""
#
#


# The key problem single-variate time-series forecasting is that: minimizing the error doesnot really provide better forecasting 
""" Time-Series Key Summary:
    # 1.
"""
