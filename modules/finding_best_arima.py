import numpy as np
import pandas as pd
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools 



df = pd.read_csv('Data_base/pivioted_fully_preprocessed_70-20.csv', infer_datetime_format = True, parse_dates=[1])
df = df.set_index(['country_region', 'year'])

def find_best_arima_model(country: str, columns: list[str]):
    analyse_df = df.loc[country, columns]
    analyse_df = analyse_df.resample('A').mean()
    arima_params = []
    mae_list = []
    rmse_list = []
    for name, series in analyse_df.items():
        if analyse_df[name].isna().mean() == 1:
            arima_params.append('a')
            mae_list.append('a')
            rmse_list.append('a')
            continue
        elif len(analyse_df[name].dropna()) < 10:
            arima_params.append('b')
            mae_list.append('b')
            rmse_list.append('b')
            continue
#        elif analyse_df[name].iloc[-5:].isna().mean() == 1:
#            arima_params.append('x')
#            continue
#        elif analyse_df[name].iloc[-5:].isna().mean() > 0.6 and np.isnan(analyse_df[name][-1]):
#            arima_params.append('c')
#            continue
        series_to_model = analyse_df[name][:analyse_df[name].last_valid_index()].ffill().dropna()
        X_train, X_test = series_to_model.iloc[:-5], series_to_model.iloc[-5:]

        best_mae = float('inf')
        best_rmse = float('inf')
        best_order = None

        values = range(1, 4)

        for p, d, q in itertools.product(values, repeat=3):
            try:
                order = (p, d, q)
                model = ARIMA(endog = X_train, order=order, freq = 'A')
                arima_fitted = model.fit()

                forecast = arima_fitted.forecast(steps=5)
                mae = mean_absolute_error(X_test, forecast)
                rmse = sqrt(mean_squared_error(X_test, forecast))

                if mae < best_mae and rmse < best_rmse:
                    best_mae = mae
                    best_rmse = rmse
                    best_order = order
            except:
                continue
        arima_params.append(best_order)
        mae_list.append(best_mae)
        rmse_list.append(best_rmse)
    return arima_params, mae_list, rmse_list