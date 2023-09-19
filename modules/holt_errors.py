import numpy as np
import pandas as pd
from math import sqrt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('Data_base/pivioted_fully_preprocessed_70-20.csv', infer_datetime_format = True, parse_dates=[1])
df = df.set_index(['country_region', 'year'])

def holt_errors(country: str, columns: list[str]):
    X_analyse = df.loc[country, columns]
    X_analyse = X_analyse.resample('A').mean()
    
    mae_list = []
    rmse_list = []

    for name, series in X_analyse.items():
        if X_analyse[name].isna().mean() == 1:
            mae_list.append('a')
            rmse_list.append('a')
            continue
        elif len(X_analyse[name].dropna()) < 10:
            mae_list.append('b')
            rmse_list.append('b')
            continue
        
        X_analyse[name] = X_analyse[name][:X_analyse[name].last_valid_index()]
        
        X_train, X_test = X_analyse[name].dropna().iloc[:-5], X_analyse[name].dropna().iloc[-5:]
        model = ExponentialSmoothing(endog = X_train, trend = 'add', seasonal=None)
        fit_model = model.fit()
        forecast = fit_model.forecast(steps=5)
        
        mae = mean_absolute_error(X_test, forecast)
        rmse = sqrt(mean_squared_error(X_test, forecast))
        
        mae_list.append(mae)
        rmse_list.append(rmse)
    return mae_list, rmse_list