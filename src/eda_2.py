
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from modules.ml_func_2 import *
from plotly.subplots import make_subplots

def eda_2_app():
    st.subheader(body="Exploratory Financial Data Analysis :chart:")
    
    file_path = './data/inputs/df_trading_input.csv'
    df = read_data(file_path)
    
    # Convertir la columna 'date' a datetime
    df['date'] = pd.to_datetime(df['date'])

    # Obtener filtros de fecha desde modules
    start_date, end_date = get_date_filters(df)

    # Filtrar datos basados en los filtros seleccionados
    filtered_df = filter_data(df, start_date, end_date)

    # Título y descripción
    st.title('Análisis del Sector Financiero')
    st.write('Esta aplicación visualiza datos del sector financiero, incluyendo precios de cierre, volumen de operaciones y análisis de sentimiento.')

    # Poner el DataFrame en un expander
    with st.expander('DataFrame'):
        st.subheader('DataFrame Filtrado')
        st.write(filtered_df)

    # Line chart para precios de cierre a lo largo del tiempo
    st.subheader('Precios de cierre a lo largo del tiempo')
    fig_close = px.line(filtered_df, x='date', y='Financial_Sector_Close', title='Precios de cierre del Sector Financiero')
    fig_close.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_close)

    # Bar chart para volumen de operaciones a lo largo del tiempo
    st.subheader('Volumen de Operaciones a lo largo del tiempo')
    for year in sorted(filtered_df['date'].dt.year.unique()):
        year_df = filtered_df[filtered_df['date'].dt.year == year]

        fig_volume = px.bar(year_df, x='date', y='Volume', title=f'Volumen de Operaciones - Año {year}', color='Volume', color_continuous_scale=[(0, "lightblue"), (1, "darkblue")])
        fig_volume.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
        st.plotly_chart(fig_volume)
        
    # Line charts para conteo de sentimientos
    st.subheader('Conteo de sentimientos a lo largo del tiempo')
    for year in sorted(filtered_df['date'].dt.year.unique()):
        year_df = filtered_df[filtered_df['date'].dt.year == year]

        fig_sentiment_year = go.Figure()

        fig_sentiment_year.add_trace(go.Scatter(
            x=year_df['date'], y=year_df['Positive'], mode='lines', name='Positivo', line=dict(color='green')
        ))

        fig_sentiment_year.add_trace(go.Scatter(
            x=year_df['date'], y=year_df['Negative'], mode='lines', name='Negativo', line=dict(color='red')
        ))

        fig_sentiment_year.add_trace(go.Scatter(
            x=year_df['date'], y=year_df['Neutral'], mode='lines', name='Neutral', line=dict(color='gray')
        ))

        fig_sentiment_year.update_layout(
            title=f'Conteo de sentimientos - Año {year}',
            xaxis_title='Fecha',
            yaxis_title='Conteo de sentimientos',
            xaxis=dict(rangeslider=dict(visible=True))
        )

        st.plotly_chart(fig_sentiment_year)

    # Line charts separados para VIX y Volatilidad Histórica
    st.subheader('Cierre del VIX y Volatilidad Histórica a lo largo del tiempo')
    for year in sorted(filtered_df['date'].dt.year.unique()):
        year_df = filtered_df[filtered_df['date'].dt.year == year]

        fig_vix_volatility_year = make_subplots(specs=[[{"secondary_y": True}]])

        fig_vix_volatility_year.add_trace(
            go.Scatter(x=year_df['date'], y=year_df['VIX_Close'], name='VIX', line=dict(color='red')),
            secondary_y=False,
        )

        fig_vix_volatility_year.add_trace(
            go.Scatter(x=year_df['date'], y=year_df['Historical_Volatility'], name='Volatilidad Histórica', line=dict(color='blue')),
            secondary_y=True,
        )

        # Ajustar el eje Y a ambas variables
        min_vix_year = year_df['VIX_Close'].min()
        max_vix_year = year_df['VIX_Close'].max()
        min_volatility_year = year_df['Historical_Volatility'].min()
        max_volatility_year = year_df['Historical_Volatility'].max()

        combined_min_year = min(min_vix_year, min_volatility_year)
        combined_max_year = max(max_vix_year, max_volatility_year)

        fig_vix_volatility_year.update_layout(
            title_text=f'Cierre del VIX y Volatilidad Histórica a lo largo del tiempo - Año {year}',
            xaxis_title='Fecha',
            yaxis_title='Valor',
            xaxis=dict(rangeslider=dict(visible=True))
        )

        fig_vix_volatility_year.update_yaxes(title_text='VIX', secondary_y=False, range=[combined_min_year, combined_max_year])
        fig_vix_volatility_year.update_yaxes(title_text='Volatilidad Histórica', secondary_y=True, range=[combined_min_year, combined_max_year])

        st.plotly_chart(fig_vix_volatility_year)
    
    # Gráficas para Alpha y Beta a lo largo del tiempo
    st.subheader('Alpha a lo largo del tiempo')
    fig_alpha = px.line(filtered_df, x='date', y='Alpha', title='Alpha a lo largo del tiempo', color_discrete_sequence=['yellow'])
    fig_alpha.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_alpha)

    st.subheader('Beta a lo largo del tiempo')
    fig_beta = px.line(filtered_df, x='date', y='Beta', title='Beta a lo largo del tiempo', color_discrete_sequence=['purple'])
    fig_beta.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    st.plotly_chart(fig_beta)

    # Heatmap de Matriz de Correlación
    st.subheader('Matriz de Correlación')
    
    filtered_df_no_index_date = filtered_df.loc[:, ~filtered_df.columns.isin(['Unnamed: 0', 'date'])] # Quitamos la fecha y el índice
    corr = filtered_df_no_index_date.corr()
    
    fig_corr = ff.create_annotated_heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale='RdBu',
        annotation_text=corr.round(2).values,
        showscale=True
    )
    
    fig_corr.update_layout(
        title_text='Matriz de Correlación',
        xaxis_title='Variables',
        yaxis_title='Variables'
    )
    
    st.plotly_chart(fig_corr)

