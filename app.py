import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Завантаження моделі (кешується, щоб не вантажити на кожен клік)
# ЗМІНИ назву файлу на свою, якщо вона інша!
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/aussie_rain.joblib")

model_dict = load_model()

model = model_dict["model"]
imputer = model_dict["imputer"]
scaler = model_dict["scaler"]
encoder = model_dict["encoder"]
numeric_cols = model_dict["numeric_cols"]
categorical_cols = model_dict["categorical_cols"]
encoded_cols = model_dict["encoded_cols"]
input_cols = model_dict["input_cols"]

# ---------------------------------------------------------
# Довідники значень для категоріальних полів
# ---------------------------------------------------------
LOCATIONS = ['Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek', 'Ballarat',
             'Bendigo', 'Brisbane', 'Cairns', 'Canberra', 'Cobar', 'CoffsHarbour', 'Dartmoor',
             'Darwin', 'GoldCoast', 'Hobart', 'Katherine', 'Launceston', 'Melbourne',
             'MelbourneAirport', 'Mildura', 'Moree', 'MountGambier', 'MountGinini', 'Newcastle',
             'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 'PearceRAAF', 'Penrith', 'Perth',
             'PerthAirport', 'Portland', 'Richmond', 'Sale', 'SalmonGums', 'Sydney',
             'SydneyAirport', 'Townsville', 'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole',
             'Watsonia', 'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera']

WIND_DIRS = ['E', 'ENE', 'ESE', 'N', 'NE', 'NNE', 'NNW', 'NW', 'S', 'SE', 'SSE', 'SSW',
             'SW', 'W', 'WNW', 'WSW']

# ---------------------------------------------------------
# Інтерфейс
# ---------------------------------------------------------
st.title("🌦️ Rain in Australia — прогноз дощу на завтра")
st.write("Заповни параметри погоди на сьогодні, щоб отримати прогноз, чи піде дощ завтра.")

with st.sidebar:
    st.header("Вхідні параметри")

    location = st.selectbox("Location", LOCATIONS)

    min_temp = st.number_input("MinTemp (°C)", value=15.0, step=0.5)
    max_temp = st.number_input("MaxTemp (°C)", value=25.0, step=0.5)
    rainfall = st.number_input("Rainfall (мм)", value=0.0, min_value=0.0, step=0.5)
    evaporation = st.number_input("Evaporation", value=5.0, min_value=0.0, step=0.5)
    sunshine = st.number_input("Sunshine (год)", value=7.0, min_value=0.0, max_value=24.0, step=0.5)

    wind_gust_dir = st.selectbox("WindGustDir", WIND_DIRS)
    wind_gust_speed = st.number_input("WindGustSpeed (км/год)", value=40.0, step=1.0)
    wind_dir_9am = st.selectbox("WindDir9am", WIND_DIRS)
    wind_dir_3pm = st.selectbox("WindDir3pm", WIND_DIRS)
    wind_speed_9am = st.number_input("WindSpeed9am", value=15.0, step=1.0)
    wind_speed_3pm = st.number_input("WindSpeed3pm", value=20.0, step=1.0)

    humidity_9am = st.slider("Humidity9am (%)", 0, 100, 70)
    humidity_3pm = st.slider("Humidity3pm (%)", 0, 100, 50)

    pressure_9am = st.number_input("Pressure9am (hPa)", value=1015.0, step=0.5)
    pressure_3pm = st.number_input("Pressure3pm (hPa)", value=1013.0, step=0.5)

    cloud_9am = st.slider("Cloud9am (хмарність, 0-9 часток неба)", 0, 9, 4)
    cloud_3pm = st.slider("Cloud3pm (хмарність, 0-9 часток неба)", 0, 9, 4)

    temp_9am = st.number_input("Temp9am (°C)", value=18.0, step=0.5)
    temp_3pm = st.number_input("Temp3pm (°C)", value=23.0, step=0.5)

    rain_today = st.selectbox("RainToday", ["No", "Yes"])

# ---------------------------------------------------------
# Формуємо DataFrame з одним рядком у порядку input_cols
# ---------------------------------------------------------
def build_input_df():
    raw = {
        "Location": location,
        "MinTemp": min_temp,
        "MaxTemp": max_temp,
        "Rainfall": rainfall,
        "Evaporation": evaporation,
        "Sunshine": sunshine,
        "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed,
        "WindDir9am": wind_dir_9am,
        "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am,
        "WindSpeed3pm": wind_speed_3pm,
        "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm,
        "Pressure9am": pressure_9am,
        "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am,
        "Cloud3pm": cloud_3pm,
        "Temp9am": temp_9am,
        "Temp3pm": temp_3pm,
        "RainToday": rain_today,
    }
    return pd.DataFrame([raw])[input_cols]


def preprocess(df):
    df = df.copy()
    df[numeric_cols] = imputer.transform(df[numeric_cols])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df[encoded_cols] = encoder.transform(df[categorical_cols])
    return df[numeric_cols + encoded_cols]


# ---------------------------------------------------------
# Передбачення
# ---------------------------------------------------------
if st.button("Передбачити"):
    input_df = build_input_df()
    processed_df = preprocess(input_df)

    prediction = model.predict(processed_df)[0]
    probability = model.predict_proba(processed_df)[0][list(model.classes_).index(prediction)]

    if prediction == "Yes":
        st.error(f"🌧️ Завтра ймовірно буде дощ (ймовірність: {probability:.2%})")
    else:
        st.success(f"☀️ Завтра ймовірно без дощу (ймовірність: {probability:.2%})")
