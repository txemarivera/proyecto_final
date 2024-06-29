#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st

def read_data(file_path):
    return pd.read_csv(file_path)

def get_date_filters(df):
    # Convertir min_date y max_date a Timestamp de pandas
    min_date = pd.Timestamp(df['date'].min())
    max_date = pd.Timestamp(df['date'].max())

    # Sliders de fecha en la barra lateral para seleccionar un rango de fechas
    start_date = st.sidebar.date_input("Start date",
                                       min_value=min_date.date(),
                                       max_value=max_date.date(),
                                       value=min_date.date(),
                                       key='start_date_slider')

    end_date = st.sidebar.date_input("End date",
                                     min_value=min_date.date(),
                                     max_value=max_date.date(),
                                     value=max_date.date(),
                                     key='end_date_slider')

    return start_date, end_date

def filter_data(df, start_date, end_date):
    # Filtrar el DataFrame basado en las fechas seleccionadas
    filtered_df = df[(df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))]
    return filtered_df
