## Hotel Room Sold Forecast

#### Dataset Summary

The hotel booking dataset consists of 32 columns and 119,390 rows.
This dataset is available at https://www.kaggle.com/jessemostipak/hotel-booking-demand

#### Objective

To predict the number of rooms sold using different forecasting models.
Root mean squared error (RMSE) and mean absolute percentage error (MAPE) are used to compare the forecasting accuracy between models.

#### Result

The result is as followed:

|Method|RMSE|MAPE|
|-----------|---------|--------|
Naive Method  | 5.78 | 2.46 
Simple Average Method | 32.39 | 17.83
Simple Moving Average Method  | 5.49 |  2.45
Simple Exponential Smoothing Method  | 5.57 |  2.58
Holt's Exponential Smoothing Method | 30.19 | 15.37
Holt Winters' Additive Method  | 7.50  | 3.52
Holt Winters' Additive Method |  7.36 |  3.44
Auto Regression Method | 34.31 |  5.23
Moving Average Method | 34.32  | 6.32
Auto Regression Moving Average Method | 34.32  | 5.97