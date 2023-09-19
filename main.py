import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


from modules.tfidf_model import find_most_similar
from modules.finding_best_arima import find_best_arima_model
from modules.holt_errors import holt_errors

# Загружаем датасет из файла
df = pd.read_csv('Data_base/pivioted_fully_preprocessed_70-20.csv', infer_datetime_format = True, parse_dates=[1])
df = df.set_index(['country_region', 'year'])
countries = list(df.index.get_level_values('country_region').unique())

st.title(f"Анализ базы данных 'Здоровье для всех' за 1970-2020 год")

#body = list(df.columns)
# Выводим панель, на которой предлагаем выбрать страну и ввести интересующую тему
st.sidebar.subheader('Выберите параметры')
country = st.sidebar.selectbox('Выберите страну', countries)
topic1 = st.sidebar.text_input('Введите интересуемую тему_1')
topic2 = st.sidebar.text_input('Введите интересуемую тему_2')
topic3 = st.sidebar.text_input('Введите интересуемую тему_3')
topic4 = st.sidebar.text_input('Введите интересуемую тему_4')

@st.cache_data
def find_topics(topic: str):
    return find_most_similar(topic)

similars_1 = find_topics(topic1)
similars_2 = find_topics(topic2)
similars_3 = find_topics(topic3)
similars_4 = find_topics(topic4)

columns = []
if topic1:
    column1 = st.sidebar.selectbox('Выберите показатель 1', similars_1)
    st.sidebar.write("Выбранный показатель: ", column1)
    columns.append(column1)
if topic2:#
    column2 = st.sidebar.selectbox('Выберите показатель 2', similars_2)
    st.sidebar.write("Выбранный показатель: ", column2)
    columns.append(column2)
if topic3:
    column3 = st.sidebar.selectbox('Выберите показатель 3', similars_3)
    st.sidebar.write("Выбранный показатель: ", column3)
    columns.append(column3)
if topic4:
    column4 = st.sidebar.selectbox('Выберите показатель 4', similars_4)
    st.sidebar.write("Выбранный показатель: ",  column4)
    columns.append(column4)

# Создаем 3 колонки
col1, col2, col3 = st.columns(spec = [35, 35, 30])

analyse_df = df.loc[country, columns]

def show_graphics(dataset: pd.DataFrame, column: str, country: str):
    fig, ax = plt.subplots(figsize=(30, 15))
    ax.plot(dataset.loc[:, column], lw=1, marker = 'o', markersize  = 5)
    ax.set(title=f"{column}, {country}")
    st.pyplot(fig)
git@github.com:AlekseyRY/Healthcare-HFA-DB_analysis.git

if col1.button('Показать графики', key ='button'):
    # Отображаем графики выбранных показателей
    for col, col_values in analyse_df.items():
        if analyse_df[col].isna().mean() == 1:
            st.write(f"{col}, НЕТ ДАННЫХ (вообще)")
        else:
            show_graphics(dataset = analyse_df, column=col, country=country)

if col3.button('Показать статистики'):
    # Отображаем графики ACF, PACF
    for feature, values in analyse_df.items():
        if analyse_df[feature].isna().mean() == 1:
            st.write(f"{country}, {feature}, - ДАННЫЕ ДЛЯ АНАЛИЗА ОТСУТСТВУЮТ")
            st.write("_" * 110)
            continue
        elif analyse_df[feature].isna().mean() > 0:
            values = values.dropna()
            fig, (ax_acf, ax_pacf) = plt.subplots(nrows=2, figsize=(11.5, 8), sharex=False)
            plot_acf(x=analyse_df[feature].dropna(), ax=ax_acf, title=f"Autocorrelation of {feature.title()}")         
            plot_pacf(x=analyse_df[feature].dropna(), 
                        ax=ax_pacf, 
                        lags = (analyse_df[feature].dropna().shape[0] // 2) - 1, 
                        title=f"Partial Autocorrelation of {feature.title()}"
                        )
        else:
            fig, (ax_acf, ax_pacf) = plt.subplots(nrows=2, figsize=(11.5, 8), sharex=False)
            plot_acf(x=values, ax=ax_acf, title=f"Autocorrelation of {feature.title()}")
            plot_pacf(x=values, ax=ax_pacf, title=f"Partial Autocorrelation of {feature.title()}")
        st.pyplot(fig)
    # Выводим статистики теста Дики-Фуллера
    st.write("<h3>Статистика теста Дики-Фуллера</h3>", unsafe_allow_html=True,  clear_output = False)
    for name, series in analyse_df.items():
        if analyse_df[name].nunique() == 1:
            st.write(f"{country}, {name}, - ВРЕМЕННОЙ РЯД СОСТОИТ ИЗ ОДНОГО И ТОГО ЖЕ ЗНАЧЕНИЯ")
            st.write("_" * 115)
            continue
        if analyse_df[name].isna().mean() == 1.0 :
            st.write(f"{country}, {name}, - ДАННЫЕ ДЛЯ АНАЛИЗА ОТСУТСТВУЮТ")
            st.write("_" * 115)
            continue
        else:
            try: 
                st.write(f"{country}, {name}")
                st.write(f"ADF Value (Test Statistic): {round(adfuller(analyse_df[name].dropna())[0], 3)}")
                st.write(f"p-value: {round(adfuller(analyse_df[name].dropna())[1], 3)}")
                st.write(f"Critical Values: { {key: round(values, 3) for key, values in adfuller(analyse_df[name].dropna())[4].items()} }")
            except ValueError as e:
                st.write(f"{country}, {name},- НЕДОСТАТОЧНО ДАННЫХ ДЛЯ АНАЛИЗА")
        
        if (adfuller(analyse_df[name].dropna())[1] < 0.05) and (adfuller(analyse_df[name].dropna())[0] < adfuller(analyse_df[name].dropna())[4]['5%']):
            st.write("С высокой долей вероятности ряд стационарен")
        else:
            st.write("С высокой долей вероятности ряд НЕ стационарен")
        st.write('-'*110)
    

# Выводим прогноз с Holt-Winters
if col2.button('Показать прогноз на 5 лет с Holt-Winters'):
    # Получаем ошибки для прогнозов с Holt-Winters
    mae_list, rmse_list = holt_errors(country = country, columns = columns)
    i = 0
    analyse_df = analyse_df.resample('A').mean()
    for name, series in analyse_df.items():
                if analyse_df[name].isna().mean() == 1:
                    st.write(f"{name}, НЕТ ДАННЫХ (вообще)")
                    i += 1
                    continue
                elif len(analyse_df[name].dropna()) < 10:
                    st.write(f"{name}, НЕ ХВАТАЕТ ДАННЫХ ДЛЯ ПРОГНОЗА (меньше 10 значений)")
                    i += 1
                    continue                
                elif analyse_df[name].iloc[-5:].isna().mean() == 1:
                    st.write(f"{name}, НЕТ ДАННЫХ ДЛЯ ПОСЛЕДНИХ ПЯТИ ПЕРИОДОВ")
                elif analyse_df[name].iloc[-5:].isna().mean() > 0.5:
                    st.write(f"Обратите внимание, что в последних периодах недостаточно данных для корректного\ качественного прогноза")
                # Удаляем nan значения из конца серии
                series_to_model = analyse_df[name][:analyse_df[name].last_valid_index()]
                # Создаемоем модель              
                model = ExponentialSmoothing(endog = series_to_model.ffill().dropna(), trend = 'add', seasonal=None)
                fit_model = model.fit()
                forecast = fit_model.forecast(steps=5)
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(series_to_model.ffill().dropna().index, series_to_model.ffill().dropna(), label='Исходные данные (без nan)', marker = 'o', markersize  = 5)
                ax.plot(fit_model.fittedvalues.index[1:], fit_model.fittedvalues[1:], label="Прогноз на исходные данные")
                ax.plot(forecast.index, forecast, label='Прогноз на 5 периодов')
                ax.legend()
                ax.set_title(f"{name}, Прогноз временного ряда с Holt-Winters")
                ax.set_xlabel('Год')
                ax.set_ylabel('Значение')
                st.pyplot(fig)
                st.write('test MAE: %f' % round(mae_list[i], 3))
                st.write('test RMSE: %f' % round(rmse_list[i], 3))
                i += 1


# Выводим прогноз с ARIMA
if col2.button('Показать прогноз на 5 лет с ARIMA'):
    order_list, mae_list, rmse_list = find_best_arima_model(country = country, columns = columns)
    analyse_df = analyse_df.resample('A').mean()
    i = 0
    for name, series in analyse_df.items():
        if analyse_df[name].isna().mean() == 1:
            st.write(f"{name}, НЕТ ДАННЫХ (вообще)")
            i += 1
            st.write("_" * 110)
            continue
        elif len(analyse_df[name].dropna()) < 10:
            st.write(f"{name}, СЛИШКОМ МАДО ДАННЫХ ДЛЯ ПРОГНОЗА (менее 10)")
            i += 1
            st.write("_" * 110)
            continue
        elif analyse_df[name].iloc[-5:].isna().mean() == 1:
            st.write(f"{name}, НЕТ ДАННЫХ ДЛЯ ПОСЛЕДНИХ ПЯТИ ПЕРИОДОВ")
        elif analyse_df[name].iloc[-6:].isna().mean() > 0.5:
            st.write(f"Обратите внимание, что в последних периодах может быть недостаточно данных для корректного\качественного прогноза")
        # Удаляем nan значения из конца серии
        series_to_model = analyse_df[name][:analyse_df[name].last_valid_index()].ffill().dropna()

        model = ARIMA(endog = series_to_model, order=order_list[i], freq = 'A')
        arima_fitted = model.fit()
        aic = arima_fitted.aic
        bic = arima_fitted.bic
        forecast = arima_fitted.forecast(steps=5)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(series_to_model.index, series_to_model, label='Исходные данные', marker = 'o', markersize  = 5)
        ax.plot(arima_fitted.fittedvalues.index[1:], arima_fitted.fittedvalues[1:], label='Прогноз на исходные данные')
        ax.plot(pd.date_range(start=series_to_model.index[-1], periods=5, freq='A'), forecast, label='Прогноз на 5 периодов')
        ax.legend()
        ax.set_title(f"{name}, Прогноз временного ряда с ARIMA")
        ax.set_xlabel('Год')
        ax.set_ylabel('Значение')
        st.pyplot(fig)
        st.write(f"test MAE: {round(mae_list[i], 3)}")
        st.write(f"test RMSE: {round(mae_list[i], 3)}")
        st.write(f"order: {order_list[i]}")
        st.write("_" * 110)
        i += 1



button_html = """
    <div style="position: fixed; left: 20px; top: 20px;">
        <a href="https://youtu.be/dQw4w9WgXcQ" target="_blank">
            <button style="background-color: red; color: white; border-radius: 50%; 
                        width: 40px; height: 40px; font-size: 18px; border: none;">
                RR
            </button>
        </a>
    </div>
"""

st.sidebar.markdown(button_html, unsafe_allow_html=True)