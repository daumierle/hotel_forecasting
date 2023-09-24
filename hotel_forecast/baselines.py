import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pylab import rcParams

import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

from scipy.stats import boxcox
from scipy.special import inv_boxcox
import seaborn as sns
import warnings

pd.options.mode.chained_assignment = 'raise'
warnings.filterwarnings("ignore")


def data_loader(path):
    df = pd.read_csv(path)

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
    # daily['occupancy'] = daily['rooms'] / 200

    # Split data into training set and validation set
    train = daily['2015-08-01':'2017-05-30']
    valid = daily['2017-05-31':'2017-08-31']

    return daily, train, valid


# Plot time series data
def data_viz(train):
    train['rooms'].plot(figsize=(12, 4))
    plt.legend(loc='best')
    plt.title('Daily Rooms Sold')
    plt.ylim([0.0, 1.0])
    plt.close()

    # Outlier Detection
    fig = plt.subplots(figsize=(12, 2))
    ax = sns.boxplot(x=train['rooms'], whis=1.5)
    plt.close()

    # Time Series Decomposition
    rcParams['figure.figsize'] = 12, 8
    add_decomp = sm.tsa.seasonal_decompose(train['rooms'], model='additive')
    add_decomp.plot()
    plt.close()
    multi_decomp = sm.tsa.seasonal_decompose(train['rooms'], model='multiplicative')
    multi_decomp.plot()
    plt.close()

    train_len = train.shape[0]

    return train_len


'''Simple Time Series Method'''


# Naive Method
def naive(train, valid):
    y_hat_naive = valid.copy()
    y_hat_naive['naive_forecast'] = train['rooms'][-1]

    plt.figure(figsize=(12, 4))
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_naive['naive_forecast'], label='Naive Forecast')
    plt.legend(loc='best')
    plt.title('Naive Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_naive['naive_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(valid['rooms'] - y_hat_naive['naive_forecast']) / valid['rooms']) * 100, 2)
    results = pd.DataFrame({'Method': ['Naive Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return results


# Simple Average Method
def simple_average(train, valid):
    y_hat_avg = valid.copy()
    y_hat_avg['avg_forecast'] = train['rooms'].mean()

    plt.figure(figsize=(12, 4))
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_avg['avg_forecast'], label='Simple Average Forecast')
    plt.legend(loc='best')
    plt.title('Simple Average Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_avg['avg_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(valid['rooms'] - y_hat_avg['avg_forecast']) / valid['rooms']) * 100, 2)
    tempresults0 = pd.DataFrame({'Method': ['Simple Average Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults0


# Simple Moving Average
def simple_moving_avg(daily, train, valid, train_len):
    y_hat_sma = daily.copy()
    ma_window = 12
    y_hat_sma['sma_forecast'] = daily['rooms'].rolling(ma_window).mean()
    y_hat_sma.loc[train_len:, ['sma_forecast']] = y_hat_sma['sma_forecast'][train_len - 1]

    plt.figure(figsize=(12, 4))
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_sma['sma_forecast'], label='Simple Moving Average Forecast')
    plt.legend(loc='best')
    plt.title('Simple Moving Average Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_sma['sma_forecast'][train_len:])).round(2)
    mape = np.round(
        np.mean(np.abs(valid['rooms'] - y_hat_sma['sma_forecast'][train_len:]) / valid['rooms']) * 100, 2)
    tempresults1 = pd.DataFrame({'Method': ['Simple Moving Average Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults1


'''Exponential Smoothing Methods'''


# Simple Exponential Smoothing
def simple_expo(train, valid):
    model = SimpleExpSmoothing(train['rooms'])
    model_fit = model.fit(smoothing_level=0.2,
                          optimized=False)  # Lower smoothing level gives more weight to farther observations
    # print(model_fit.params)
    y_hat_ses = valid.copy()
    y_hat_ses['ses_forecast'] = model_fit.forecast(len(valid))
    y_hat_ses['ses_forecast'].fillna(y_hat_ses['ses_forecast'].mean(), inplace=True)

    plt.figure(figsize=(12, 4))
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_ses['ses_forecast'], label='Simple Exponential Smoothing Forecast')
    plt.legend(loc='best')
    plt.title('Simple Exponential Smoothing Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_ses['ses_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(valid['rooms'] - y_hat_ses['ses_forecast']) / valid['rooms']) * 100, 2)
    tempresults2 = pd.DataFrame({'Method': ['Simple Exponential Smoothing Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults2


# Holt's Method with Trend
def holt_trend(train, valid):
    model = ExponentialSmoothing(np.asarray(train['rooms']), seasonal_periods=12, trend='additive', seasonal=None)
    model_fit = model.fit(smoothing_level=0.1, smoothing_slope=0.001, optimized=False)
    # print(model_fit.params)
    y_hat_holt = valid.copy()
    y_hat_holt['holt_forecast'] = model_fit.forecast(len(valid))

    plt.figure(figsize=(12, 4))
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_holt['holt_forecast'], label='Holt\'s Exponential Smoothing Forecast')
    plt.legend(loc='best')
    plt.title('Holt\'s Exponential Smoothing Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_holt['holt_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(valid['rooms'] - y_hat_holt['holt_forecast']) / valid['rooms']) * 100, 2)
    tempresults3 = pd.DataFrame({'Method': ['Holt\'s Exponential Smoothing Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults3


# Holt Winters' Additive Method with Trend and Seasonality
def holt_add(train, valid):
    model = ExponentialSmoothing(np.asarray(train['rooms']), seasonal_periods=12, seasonal='add', trend='add')
    model_fit = model.fit(optimized=True)
    # print(model_fit.params)
    y_hat_hwa = valid.copy()
    y_hat_hwa['hwa_forecast'] = model_fit.forecast(len(valid))

    plt.figure(figsize=(12, 4))
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_hwa['hwa_forecast'], label='Holt Winters\' Additive Forecast')
    plt.legend(loc='best')
    plt.title('Holt Winters\' Additive Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_hwa['hwa_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(valid['rooms'] - y_hat_hwa['hwa_forecast']) / valid['rooms']) * 100, 2)
    tempresults4 = pd.DataFrame({'Method': ['Holt Winters\' Additive Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults4


# Holt Winters' Multiplicative Method with Trend and Seasonality
def holt_mult(train, valid):
    model = ExponentialSmoothing(np.asarray(train['rooms']), seasonal_periods=12, seasonal='mul', trend='add')
    model_fit = model.fit(optimized=True)
    # print(model_fit.params)
    y_hat_hwm = valid.copy()
    y_hat_hwm['hwm_forecast'] = model_fit.forecast(len(valid))
    y_hat_hwm['hwm_forecast'].isnull().sum()

    plt.figure(figsize=(12, 4))
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_hwm['hwm_forecast'], label='Holt Winters\' Multiplicative Forecast')
    plt.legend(loc='best')
    plt.title('Holt Winters\' Multiplicative Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_hwm['hwm_forecast'])).round(2)
    mape = np.round(np.mean(np.abs(valid['rooms'] - y_hat_hwm['hwm_forecast']) / valid['rooms']) * 100, 2)
    tempresults5 = pd.DataFrame({'Method': ['Holt Winters\' Additive Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults5


'''Auto Regressive Methods'''


# Stationary vs. Non-Stationary Time Series
def stationary_test(daily):
    daily['rooms'].plot(figsize=(12, 4))
    plt.legend(loc='best')
    plt.title('Daily Rooms Sold')
    plt.close()
    # Augmented Dickey-Fuller (ADF) Test
    adf_test = adfuller(daily['rooms'])
    # print('ADF Statistic: %f' %adf_test[0])
    # print('Critical Value at 0.05: %.2f' %adf_test[4]['5%'])
    p_value_adf = adf_test[1]
    # print('p-value: %f' %p_value_adf)
    # Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test
    kpss_test = kpss(daily['rooms'])
    # print('KPSS Statistic: %f' % kpss_test[0])
    # print('Critical Value at 0.05: %.2f' % kpss_test[3]['5%'])
    p_value_kpss = kpss_test[1]
    # print('p-value: %f' %p_value_kpss)
    # The p-value (0.064) is greater than 0.05 -> fail to reject null hypothesis -> time series is stationary

    return p_value_adf, p_value_kpss


# Box-Cox Transformation to make variance constant
def boxcox_trans(daily, train_len):
    data_boxcox, fitted_lambda = boxcox(daily['rooms'])
    plt.figure(figsize=(12, 4))
    plt.plot(data_boxcox, label='After Box-Cox Transformation')
    plt.legend(loc='best')
    plt.title('After Box Cox Transformation')
    plt.close()
    # print(f"Lambda value used for Transformation: {fitted_lambda}")
    plt.close()

    train_boxcox = data_boxcox[:train_len]
    valid_boxcox = data_boxcox[train_len:]

    return data_boxcox, train_boxcox, valid_boxcox, fitted_lambda


# Differencing to remove seasonality
def differencing(daily, data_boxcox):
    data_boxcox_diff = pd.Series(
        pd.Series(data_boxcox, index=daily.index) - pd.Series(data_boxcox, index=daily.index).shift(360),
        daily.index).dropna()
    plt.figure(figsize=(12, 4))
    plt.plot(data_boxcox_diff, label='After Box Cox Transformation and Differencing')
    plt.legend(loc='best')
    plt.title('After Box Cox Transformation and Differencing')
    plt.close()
    fig, ax = plt.subplots(1, 2)
    sns.distplot(daily['rooms'], hist=False, kde=True,
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

    train_boxcox_diff = data_boxcox_diff[:len(data_boxcox_diff) - len(valid)]
    valid_boxcox_diff = data_boxcox_diff[len(data_boxcox_diff) - len(valid):]

    return data_boxcox_diff, train_boxcox_diff, valid_boxcox_diff


# Auto Regression (AR) Method
def auto_regress(data_boxcox_diff, train_boxcox_diff, data_boxcox, daily, train, valid, fitted_lambda):
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
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_ar['ar_forecast'][valid.index.min():], label='Auto Regression Forecast')
    plt.legend(loc='best')
    plt.title('Auto Regression Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_ar['ar_forecast'][valid.index.min():])).round(2)
    mape = np.round(
        np.mean(np.abs(valid['rooms'] - y_hat_ar['ar_forecast'][valid.index.min()]) / valid['rooms']) * 100, 2)
    tempresults6 = pd.DataFrame({'Method': ['Auto Regression Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults6


# Moving Average (MA) Method
def moving_avg(data_boxcox_diff, train_boxcox_diff, data_boxcox, daily, train, valid, fitted_lambda):
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
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_ma['ma_forecast'][valid.index.min():], label='Moving Average Forecast')
    plt.legend(loc='best')
    plt.title('Moving Average Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_ma['ma_forecast'][valid.index.min():])).round(2)
    mape = np.round(
        np.mean(np.abs(valid['rooms'] - y_hat_ma['ma_forecast'][valid.index.min()]) / valid['rooms']) * 100, 2)
    tempresults7 = pd.DataFrame({'Method': ['Moving Average Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults7


# Auto regression Moving Average (ARMA) Method
def arma(data_boxcox_diff, train_boxcox_diff, data_boxcox, daily, train, valid, fitted_lambda):
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
    plt.plot(train['rooms'], label='Train')
    plt.plot(valid['rooms'], label='Valid')
    plt.plot(y_hat_arma['arma_forecast'][valid.index.min():], label='Auto Regression Moving Average Forecast')
    plt.legend(loc='best')
    plt.title('Auto Regression Moving Average Method')
    plt.close()

    rmse = np.sqrt(mean_squared_error(valid['rooms'], y_hat_arma['arma_forecast'][valid.index.min():])).round(2)
    mape = np.round(
        np.mean(np.abs(valid['rooms'] - y_hat_arma['arma_forecast'][valid.index.min()]) / valid['rooms']) * 100,
        2)
    tempresults8 = pd.DataFrame({'Method': ['Auto Regression Moving Average Method'], 'RMSE': [rmse], 'MAPE': [mape]})

    return tempresults8


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path where data is stored.")
    args = parser.parse_args()

    daily, train, valid = data_loader(args.data_path)
    train_len = data_viz(train)
    data_boxcox, train_boxcox, valid_boxcox, fitted_lambda = boxcox_trans(daily, train_len)
    data_boxcox_diff, train_boxcox_diff, valid_boxcox_diff = differencing(daily, data_boxcox)

    final_results = naive(train, valid)
    final_results = final_results.append(simple_average(train, valid), ignore_index=True)
    final_results = final_results.append(simple_moving_avg(daily, train, valid, train_len), ignore_index=True)
    final_results = final_results.append(simple_expo(train, valid), ignore_index=True)
    final_results = final_results.append(holt_trend(train, valid), ignore_index=True)
    final_results = final_results.append(holt_add(train, valid), ignore_index=True)
    final_results = final_results.append(holt_mult(train, valid), ignore_index=True)
    final_results = final_results.append(auto_regress(data_boxcox_diff, train_boxcox_diff, data_boxcox, daily, train,
                                                      valid, fitted_lambda), ignore_index=True)
    final_results = final_results.append(moving_avg(data_boxcox_diff, train_boxcox_diff, data_boxcox, daily, train,
                                                    valid, fitted_lambda), ignore_index=True)
    final_results = final_results.append(arma(data_boxcox_diff, train_boxcox_diff, data_boxcox, daily, train, valid,
                                              fitted_lambda), ignore_index=True)

    print(final_results)
