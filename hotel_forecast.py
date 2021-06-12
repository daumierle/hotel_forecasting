import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import seaborn as sns
import warnings
import torch
import torch.nn as nn
import datetime

pd.options.mode.chained_assignment = 'raise'
warnings.filterwarnings("ignore")

df = pd.read_csv('C:\\Users\\Dat\\Downloads\\CSV + JSON files\\hotel_bookings.csv')

'''Data Prep'''

df['arrival_date_month'] = pd.to_datetime(df.arrival_date_month, format='%B').dt.month
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].map(str) + '/' + df['arrival_date_month'].map(str) + '/' + df['arrival_date_day_of_month'].map(str))
df['length_of_stay'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']

resort = df.loc[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 0)]
resort = resort.loc[resort.index.repeat(resort.length_of_stay)]
resort['i'] = resort.groupby(resort.index).cumcount() + 1
resort['arrival_date'] += pd.TimedeltaIndex(resort['i'], unit='D')
resort['month_id'] = resort['arrival_date'].map(str).str.slice(stop=7)

resort_final = resort.loc[:, ['month_id', 'arrival_date', 'length_of_stay']]
resort_final = resort_final.rename(columns={'arrival_date': 'inhouse_date'})
resort_final['length_of_stay'] = 1

daily = resort_final.loc[(resort_final['month_id'] != '2015-07') & (resort_final['month_id'] != '2017-09')].groupby(
    'inhouse_date').sum()
daily = daily.rename(columns={'length_of_stay': 'rooms'})
daily['occupancy'] = daily['rooms'] / 200

# Split data into training set and validation set
train = daily['2015-08-01':'2017-05-30']
valid = daily['2017-05-31':'2017-08-31']

# Plot time series data
train['occupancy'].plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Occupancy Percentage')
plt.ylim([0.0, 1.0])
plt.close()
"""
# Outlier Detection
fig = plt.subplots(figsize=(12, 2))
ax = sns.boxplot(x=train['occupancy'], whis=1.5)
plt.close()

# Time Series Decomposition
rcParams['figure.figsize'] = 12, 8
add_decomp = sm.tsa.seasonal_decompose(train['occupancy'], model='additive')
add_decomp.plot()
plt.close()
multi_decomp = sm.tsa.seasonal_decompose(train['occupancy'], model='multiplicative')
multi_decomp.plot()
plt.close()

train_len = train.shape[0]

'''Simple Time Series Method'''

# Naive Method
y_hat_naive = valid.copy()
y_hat_naive['naive_forecast'] = train['occupancy'][-1]
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_naive['naive_forecast'], label='Naive Forecast')
plt.legend(loc='best')
plt.title('Naive Method')
plt.close()
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_naive['naive_forecast'])).round(2)
mape = np.round(np.mean(np.abs(valid['occupancy'] - y_hat_naive['naive_forecast']) / valid['occupancy']) * 100, 2)
results = pd.DataFrame({'Method': ['Naive Method'], 'RMSE': [rmse], 'MAPE': [mape]})

# Simple Average Method
y_hat_avg = valid.copy()
y_hat_avg['avg_forecast'] = train['occupancy'].mean()
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_avg['avg_forecast'], label='Simple Average Forecast')
plt.legend(loc='best')
plt.title('Simple Average Method')
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_avg['avg_forecast'])).round(2)
mape = np.round(np.mean(np.abs(valid['occupancy'] - y_hat_avg['avg_forecast']) / valid['occupancy']) * 100, 2)
tempresults = pd.DataFrame({'Method': ['Simple Average Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])
plt.close()

# Simple Moving Average
y_hat_sma = daily.copy()
ma_window = 12
y_hat_sma['sma_forecast'] = daily['occupancy'].rolling(ma_window).mean()
y_hat_sma.loc[train_len:, ['sma_forecast']] = y_hat_sma['sma_forecast'][train_len - 1]
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_sma['sma_forecast'], label='Simple Moving Average Forecast')
plt.legend(loc='best')
plt.title('Simple Moving Average Method')
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_sma['sma_forecast'][train_len:])).round(2)
mape = np.round(
    np.mean(np.abs(valid['occupancy'] - y_hat_sma['sma_forecast'][train_len:]) / valid['occupancy']) * 100, 2)
tempresults = pd.DataFrame({'Method': ['Simple Moving Average Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])
plt.close()

'''Exponential Smoothing Methods'''

# Simple Exponential Smoothing
model = SimpleExpSmoothing(train['occupancy'])
model_fit = model.fit(smoothing_level=0.2,
                      optimized=False)  # Lower smoothing level gives more weight to farther observations
# print(model_fit.params)
y_hat_ses = valid.copy()
y_hat_ses['ses_forecast'] = model_fit.forecast(len(valid))
y_hat_ses['ses_forecast'].fillna(y_hat_ses['ses_forecast'].mean(), inplace=True)
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_ses['ses_forecast'], label='Simple Exponential Smoothing Forecast')
plt.legend(loc='best')
plt.title('Simple Exponential Smoothing Method')
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_ses['ses_forecast'])).round(2)
mape = np.round(np.mean(np.abs(valid['occupancy'] - y_hat_ses['ses_forecast']) / valid['occupancy']) * 100, 2)
tempresults = pd.DataFrame({'Method': ['Simple Exponential Smoothing Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])
plt.close()

# Holt's Method with Trend
model = ExponentialSmoothing(np.asarray(train['occupancy']), seasonal_periods=12, trend='additive', seasonal=None)
model_fit = model.fit(smoothing_level=0.1, smoothing_slope=0.001, optimized=False)
# print(model_fit.params)
y_hat_holt = valid.copy()
y_hat_holt['holt_forecast'] = model_fit.forecast(len(valid))
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s Exponential Smoothing Forecast')
plt.legend(loc='best')
plt.title('Holt\'s Exponential Smoothing Method')
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_holt['holt_forecast'])).round(2)
mape = np.round(np.mean(np.abs(valid['occupancy'] - y_hat_holt['holt_forecast']) / valid['occupancy']) * 100, 2)
tempresults = pd.DataFrame({'Method': ['Holt\'s Exponential Smoothing Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])
plt.close()

# Holt Winters' Additive Method with Trend and Seasonality
model = ExponentialSmoothing(np.asarray(train['occupancy']), seasonal_periods=12, seasonal='add', trend='add')
model_fit = model.fit(optimized=True)
# print(model_fit.params)
y_hat_hwa = valid.copy()
y_hat_hwa['hwa_forecast'] = model_fit.forecast(len(valid))
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_hwa['hwa_forecast'], label='Holt Winters\' Additive Forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Additive Method')
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_hwa['hwa_forecast'])).round(2)
mape = np.round(np.mean(np.abs(valid['occupancy'] - y_hat_hwa['hwa_forecast']) / valid['occupancy']) * 100, 2)
tempresults = pd.DataFrame({'Method': ['Holt Winters\' Additive Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])
plt.close()

# Holt Winters' Multiplicative Method with Trend and Seasonality
model = ExponentialSmoothing(np.asarray(train['occupancy']), seasonal_periods=12, seasonal='mul', trend='add')
model_fit = model.fit(optimized=True)
# print(model_fit.params)
y_hat_hwm = valid.copy()
y_hat_hwm['hwm_forecast'] = model_fit.forecast(len(valid))
y_hat_hwm['hwm_forecast'].isnull().sum()
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_hwa['hwa_forecast'], label='Holt Winters\' Multiplicative Forecast')
plt.legend(loc='best')
plt.title('Holt Winters\' Multiplicative Method')
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_hwm['hwm_forecast'])).round(2)
mape = np.round(np.mean(np.abs(valid['occupancy'] - y_hat_hwm['hwm_forecast']) / valid['occupancy']) * 100, 2)
tempresults = pd.DataFrame({'Method': ['Holt Winters\' Additive Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])
plt.close()

'''Auto Regressive Methods'''

# Stationary vs. Non-Stationary Time Series
daily['occupancy'].plot(figsize=(12, 4))
plt.legend(loc='best')
plt.title('Occupancy Percentage')
plt.close()
# Augmented Dickey-Fuller (ADF) Test
adf_test = adfuller(daily['occupancy'])
# print('ADF Statistic: %f' %adf_test[0])
# print('Critical Value at 0.05: %.2f' %adf_test[4]['5%'])
# print('p-value: %f' %adf_test[1])
# Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
kpss_test = kpss(daily['occupancy'])
# print('KPSS Statistic: %f' % kpss_test[0])
# print('Critical Value at 0.05: %.2f' % kpss_test[3]['5%'])
# print('p-value: %f' % kpss_test[1])
# The p-value (0.064) is greater than 0.05 -> fail to reject null hypothesis -> time series is stationary

# Box-Cox Transformation to make variance constant
data_boxcox, fitted_lambda = boxcox(daily['occupancy'])
plt.figure(figsize=(12, 4))
plt.plot(data_boxcox, label='After Box-Cox Transformation')
plt.legend(loc='best')
plt.title('After Box Cox Transformation')
plt.close()
# print(f"Lambda value used for Transformation: {fitted_lambda}")
plt.close()

# Differencing to remove seasonality
data_boxcox_diff = pd.Series(
    pd.Series(data_boxcox, index=daily.index) - pd.Series(data_boxcox, index=daily.index).shift(360),
    daily.index).dropna()
plt.figure(figsize=(12, 4))
plt.plot(data_boxcox_diff, label='After Box Cox Transformation and Differencing')
plt.legend(loc='best')
plt.title('After Box Cox Transformation and Differencing')
plt.close()
fig, ax = plt.subplots(1, 2)
sns.distplot(daily['occupancy'], hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 2},
             label="Before Transformation", color="green", ax=ax[0])

sns.distplot(data_boxcox_diff, hist=False, kde=True,
             kde_kws={'shade': True, 'linewidth': 2},
             label="After Transformation", color="green", ax=ax[1])
plt.legend(loc="upper right")
fig.set_figheight(5)
fig.set_figwidth(10)
plt.close()
# p-value for KPSS Test is 0.1 > 0.05 --> time series is stationary

# Autocorrelation Function (ACF)
plt.figure(figsize=(12, 4))
plot_acf(data_boxcox_diff, ax=plt.gca(), lags=25)
plt.close()
# Partial Autocorrelation Function (PACF)
plt.figure(figsize=(12, 4))
plot_pacf(data_boxcox_diff, ax=plt.gca(), lags=25)
plt.close()

train_boxcox = data_boxcox[:train_len]
valid_boxcox = data_boxcox[train_len:]
train_boxcox_diff = data_boxcox_diff[:len(data_boxcox_diff) - len(valid)]
valid_boxcox_diff = data_boxcox_diff[len(data_boxcox_diff) - len(valid):]

# Auto Regression (AR) Method
model = ARIMA(train_boxcox_diff, order=(1, 0, 0))
model_fit = model.fit()
# print(model_fit.params)
y_hat_ar = data_boxcox_diff.copy()
y_hat_ar['ar_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox_diff'] + data_boxcox[:402]
y_hat_ar['ar_forecast_boxcox'][360:] += data_boxcox[:42]
y_hat_ar['ar_forecast_boxcox'] = y_hat_ar['ar_forecast_boxcox'].append(
    pd.Series(data_boxcox, index=daily.index)[:360]).sort_index()
y_hat_ar['ar_forecast'] = inv_boxcox(y_hat_ar['ar_forecast_boxcox'], fitted_lambda)
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_ar['ar_forecast'][valid.index.min():], label='Auto Regression Forecast')
plt.legend(loc='best')
plt.title('Auto Regression Method')
plt.close()
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_ar['ar_forecast'][valid.index.min():])).round(2)
mape = np.round(
    np.mean(np.abs(valid['occupancy'] - y_hat_ar['ar_forecast'][valid.index.min()]) / valid['occupancy']) * 100, 2)
tempresults = pd.DataFrame({'Method': ['Auto Regression Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])

# Moving Average (MA) Method
model = ARIMA(train_boxcox_diff, order=(0, 0, 2))
model_fit = model.fit()
# print(model_fit.params)
y_hat_ma = data_boxcox_diff.copy()
y_hat_ma['ma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(), data_boxcox_diff.index.max())
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox_diff'] + data_boxcox[:402]
y_hat_ma['ma_forecast_boxcox'][360:] += data_boxcox[:42]
y_hat_ma['ma_forecast_boxcox'] = y_hat_ma['ma_forecast_boxcox'].append(
    pd.Series(data_boxcox, index=daily.index)[:360]).sort_index()
y_hat_ma['ma_forecast'] = inv_boxcox(y_hat_ma['ma_forecast_boxcox'], fitted_lambda)
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_ma['ma_forecast'][valid.index.min():], label='Moving Average Forecast')
plt.legend(loc='best')
plt.title('Moving Average Method')
plt.close()
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_ma['ma_forecast'][valid.index.min():])).round(2)
mape = np.round(
    np.mean(np.abs(valid['occupancy'] - y_hat_ma['ma_forecast'][valid.index.min()]) / valid['occupancy']) * 100, 2)
tempresults = pd.DataFrame({'Method': ['Moving Average Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])

# Auto regression Moving Average (ARMA) Method
model = ARIMA(train_boxcox_diff, order=(1, 0, 1))
model_fit = model.fit()
# print(model_fit.params)
y_hat_arma = data_boxcox_diff.copy()
y_hat_arma['arma_forecast_boxcox_diff'] = model_fit.predict(data_boxcox_diff.index.min(),
                                                            data_boxcox_diff.index.max())
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox_diff'] + data_boxcox[:402]
y_hat_arma['arma_forecast_boxcox'][360:] += data_boxcox[:42]
y_hat_arma['arma_forecast_boxcox'] = y_hat_arma['arma_forecast_boxcox'].append(
    pd.Series(data_boxcox, index=daily.index)[:360]).sort_index()
y_hat_arma['arma_forecast'] = inv_boxcox(y_hat_arma['arma_forecast_boxcox'], fitted_lambda)
plt.figure(figsize=(12, 4))
plt.plot(train['occupancy'], label='Train')
plt.plot(valid['occupancy'], label='Valid')
plt.plot(y_hat_arma['arma_forecast'][valid.index.min():], label='Auto Regression Moving Average Forecast')
plt.legend(loc='best')
plt.title('Auto Regression Moving Average Method')
plt.close()
rmse = np.sqrt(mean_squared_error(valid['occupancy'], y_hat_arma['arma_forecast'][valid.index.min():])).round(2)
mape = np.round(
    np.mean(np.abs(valid['occupancy'] - y_hat_arma['arma_forecast'][valid.index.min()]) / valid['occupancy']) * 100,
    2)
tempresults = pd.DataFrame({'Method': ['Auto Regression Moving Average Method'], 'RMSE': [rmse], 'MAPE': [mape]})
results = pd.concat([results, tempresults])
"""
# LSTM Method
data = torch.FloatTensor(train['occupancy'].values.astype(float)).reshape(-1)
train_window = 365

def create_inout_sequence(input_data, tw):
    inout_seq = []
    for i in range(len(input_data) - tw):
        train_seq = input_data[i : i+tw]
        train_label = input_data[i+tw : i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

train_inout_seq = create_inout_sequence(data, train_window)

# class LSTM(nn.Module):
#     def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
#         super().__init__()
#         self.hidden_layer_size = hidden_layer_size
#         self.lstm = nn.LSTM(input_size, hidden_layer_size)
#         self.linear = nn.Linear(hidden_layer_size, output_size)
#         self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
#                             torch.zeros(1,1,self.hidden_layer_size))
#
#     def forward(self, input_seq):
#         lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
#         predictions = self.linear(lstm_out.view(len(input_seq), -1))
#         return predictions[-1]

model = nn.LSTM(input_size=1, hidden_size=100) #LSTM()
output = nn.Linear(in_features=100, out_features=1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_size),
                             torch.zeros(1, 1, model.hidden_size))
        y_pred = model(seq.view(len(seq), 1, -1))[0]   # .view(len(seq), 1, -1)
        y_hat = output(y_pred.view(len(seq), -1))[-1]
        single_loss = loss_function(y_hat, labels)
        single_loss.backward()
        optimizer.step()

    print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = 93
test_inputs = data[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        pred = output(model(seq.view(len(seq), 1, -1))[0].view(len(seq), -1))[-1]
        test_inputs.append(pred.item())
print(test_inputs[train_window:])

predictions = np.array(test_inputs[train_window:])
x = np.array([datetime.date(2017,5,31) + datetime.timedelta(days=i) for i in range(fut_pred)])

plt.figure(figsize=(12, 4))
plt.title('Long Short-Term Memory Method')
plt.autoscale(axis='x', tight=True)
plt.plot(daily['occupancy'].iloc[:669], label='Train')
plt.plot(daily['occupancy'].iloc[669:], label='Valid')
plt.plot(x, predictions, label='LSTM')
plt.legend(loc='best')
# print(results)
plt.show()

