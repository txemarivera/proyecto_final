import base64
import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model

from modules.ml_get_financial import *
from modules.ml_get_news import *
from modules.ml_actualizar_datos import *


PAGE_CONFIG = {"page_title"             : "ULTIMATE TRADING APP",
                "page_icon"             : ":chart_with_upwards_trend:",
                "layout"                : "wide",
                "initial_sidebar_state" : "expanded"}


@st.cache_data

def read_data():
    # try:
    #     df_trading_input = pd.read_csv("./data/outputs/df_trading_out.csv")
    # except:
    df_trading_input = pd.read_csv("./data/inputs/df_trading_input.csv")
    data_news_sentiment = pd.read_csv("./data/inputs/data_news_sentiment.csv")

    return df_trading_input,data_news_sentiment

def load_models():
    # if not os.getcwd().endswith("models"):
    #     os.chdir("models")

    model_trading = load_model("./models/model_trading.keras", compile=False)
    # model_trading.compile(optimizer = "adam", loss = "mse")
    
    model_glove_lstm = load_model("./models/model_glove_lstm.keras", compile=False)
    # model_glove_lstm.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])  

    # os.chdir("..")

    return model_trading, model_glove_lstm

def download_file(df, fuel_type = "all"):

    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f"<a href='data:file/csv;base64,{b64}' download='{fuel_type}_data.csv'>Download CSV File</a>"

    return href

def actualizar_mod(last_date):
    print('Iniciando actualización')
    df_fin = get_fin(last_date)
    df_news = get_news(last_date)
    df_trading_out = actualizar_datos(df_fin, df_news)
    mensaje = "Actualización completada!. Los datos se han guardado en ./data/outputs/df_trading_out.csv"
    return mensaje, df_trading_out

def load_scaler():

    if not os.getcwd().endswith("models"):
        os.chdir("models")

    with open(file = f"X_scaler.pkl", mode = "rb") as file:
        X_scaler = pickle.load(file)

    with open(file = f"y_scaler.pkl", mode = "rb") as file:
        y_scaler = pickle.load(file)
    
    with open(file = f"scaler.pkl", mode = "rb") as file:
        scaler = pickle.load(file)

    
    os.chdir("..")
    
    return X_scaler, y_scaler, scaler

def X_y_generator(data_scaled):

    T = 3 # Segmentos
    X = list()
    y = list()

    for t in range(len(data_scaled) - T):
        
        # Toma valores de X de t en t con stride de 1
        x = data_scaled[t : t + T]
        X.append(x)
        
        # Toma los valores de t en t
        ## y_ = datos[t + T]
        y_ = data_scaled[t + T, -1]
        y.append(y_)

    # Transformamos a np.array y ajustamos las dimensiones
    # Para entrar en el modelo debe de tener 3 dimensiones
    X = np.array(X).reshape(-1, T, data_scaled.shape[1] ) ##  ndim = datos.shape[1]
    ## y = np.array(y) 
    y = np.array(y).reshape(-1, 1, 1 )

    return X,y


# Esta función llama a las funciones definidas en:
# modules.ml_get_financial: descarga los datos financieros más recientes y genera un df 
# modules.ml_get_news: descarga las noticias más recientes, ejecuta el modelo de análisis de sentimiento y crea un df 
# modules.ml_actualizar_datos: une los df's anteriores para su empleo en el modelo de trading

