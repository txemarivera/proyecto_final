import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from datetime import datetime, timedelta

from modules.ml_func import *


def ml_app():
    # 1.- Title
    st.subheader(body = "Trading Model :chart_with_upwards_trend:")
    st.markdown(body = """---""")
    # st.markdown(body = """Use the sidebar to try our model.""")
   
    # 2.- Load data
        # Dataframes
    df_trading_input,data_news_sentiment = read_data()
    df_trading_input['date'] = pd.to_datetime(df_trading_input['date'])
    min_date = df_trading_input['date'].min().to_pydatetime()
    max_date = df_trading_input['date'].max().to_pydatetime()-timedelta(days=1)
       
        # Models & Scalers
    model_trading, model_glove_lstm = load_models()
    model_trading.compile(optimizer = "adam", loss = "mse")     
    X_scaler, y_scaler, scaler = load_scaler() 
    fig_1 = False
    fig_2 = False

        # Showing input data
    datos_ = df_trading_input
    datos = datos_.drop(['date'], axis=1)
    data_scaled = scaler.transform(datos)
    df_pred = pd.DataFrame(data = df_trading_input, columns = df_trading_input.columns)
    
    col1, col2 = st.columns([1, 1])
    col1.markdown(body = "User's Input:")
    col1.dataframe(data = df_pred.iloc[:, :-1])
    col2.markdown(body = "User's Prediction:")
    col2.dataframe(data = df_pred.iloc[:, -1])

    # 4.- Prediction
    # 4.1.- "One - Step Predictions"

    X,y = X_y_generator(data_scaled)
      
    len_train = int(0.8*len(datos))        
    y_pred = list()
    i = len_train
    validation_target = y[len_train:]

    # Definir variable de estado para guardar la figura
    if 'fig_1' not in st.session_state:
        st.session_state.fig_1 = None

    if st.button(label = "Run Model - One Step", key = "submit1"):  

        while len(y_pred) < len(validation_target):
            
            # Predice el siguiente valor de X[i]
            p = model_trading.predict((X[i]).reshape(1, -1, data_scaled.shape[1]))
            i += 1
            y_pred.append(p)

        y_pred = np.array(y_pred)
        validation_predictions = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        validation_target = y_scaler.inverse_transform(validation_target.reshape(-1, 1))    
    
    # 4.1.1.- Grafico 1
    # Definir vbles
        yy =  y_scaler.inverse_transform(y.reshape(-1, 1))
        yy = yy.flatten()
        validation_p = validation_predictions.flatten()     
        espacio = len(yy[:len_train]) 

        # Crear un DataFrame para Plotly Express
        data_graf_1 = {
            'Index': np.concatenate((np.arange(len(yy)), np.arange(espacio, espacio + len(validation_p)))),
            'Value': np.concatenate((yy, validation_p)),
            'Type': ['forecast target'] * len(yy) + ['forecast prediction'] * len(validation_p)
        }
        df_graf_1 = pd.DataFrame(data_graf_1)

        # Crear el gráfico usando Plotly Express
        st.session_state.fig_1 = px.line(df_graf_1, x='Index', y='Value', color='Type', 
                    labels={'Value': 'USD$', 'Index': 'Time (days)'}, 
                    title='One-Step Predictions. XLF. Financial Select Sector. Forecast vs Prediction',
                    color_discrete_map={'forecast target': 'red', 'forecast prediction': 'blue'})
        st.session_state.fig_1.update_layout(xaxis=dict(rangeslider=dict(visible=True)))                
    
    # Plot
    if st.session_state.fig_1:    
        st.plotly_chart(figure_or_data = st.session_state.fig_1, use_container_width = True)

    # 4.2.- "N - Steps Predictions"
    #    User Data Input - Display
    # Slider sidebar - steps selection
    steps_number = st.sidebar.slider(label =  "Select number of steps",
                        min_value   =  1,
                        max_value   =  300,
                        value       =  1,  # Valor inicial
                        step        =  1)  

    if 'fig_2' not in st.session_state:
        st.session_state.fig_2 = None

    if st.button(label = "Run Model - N Steps", key = "submit2"):  

        validation_predictions_2 = list()
        last_x = X[len_train]

        for i in range(steps_number):
            if len(validation_predictions_2) < len(validation_target):
            
                # En la primera iteración predice el siguiente valor de usando X
                # En las siguientes iteraciones usa el valor predicho anterior para predecir el siguiente
                p = model_trading.predict((X[i]).reshape(1, -1, datos.shape[1]))[0, 0]
                # p = model_trading.predict((X[i+len_train]).reshape(1, -1, datos.shape[1]))[0, 0]
                validation_predictions_2.append(p)
                
                            
                # Desplaza los elementos en last_x hacia atrás (en esta caso desplazamos toda la fila), dejando el primer elemento al final
                last_x = np.roll(last_x, -1, axis=0)
                
                # Cambia el último elemento a la predicción
                # en este caso se cambia solo el valor a predecir y se mantienen invariables los últimos valores de los demás parámetros
                # estos valores (análisis de sentiniento, VIX, alpha, beta...) serán los del último día conocido e iguales para los n_days_pred de la predicción
                last_x[-1][datos.shape[1]-1] = p
    
        # 4.1.2.- Grafico 2
        # Definir vbles

        yy =  y_scaler.inverse_transform(y.reshape(-1, 1))
        yy = yy.flatten()
        validation_p_2 =  y_scaler.inverse_transform((np.array(validation_predictions_2).reshape(-1, 1)))     
        validation_p_2 = validation_p_2.flatten()
        espacio = len(yy[:len_train]) 

        # Crear un DataFrame para Plotly Express
        data_graf_2 = {
            'Index': np.concatenate((np.arange(len(yy)), np.arange(espacio, espacio + len(validation_p_2)))),
            'Value': np.concatenate((yy, validation_p_2)),
            'Type': ['forecast target'] * len(yy) + ['forecast prediction'] * len(validation_p_2)
        }
        df_graf_2 = pd.DataFrame(data_graf_2)
        # Crear el gráfico usando Plotly Express
        st.session_state.fig_2 = px.line(df_graf_2, x='Index', y='Value', color='Type', 
                    labels={'Value': 'USD$', 'Index': 'Time (days)'}, 
                    title='N-Steps Predictions. XLF. Financial Select Sector. Forecast vs Prediction',
                    color_discrete_map={'forecast target': 'red', 'forecast prediction': 'blue'})
        st.session_state.fig_2.update_layout(xaxis=dict(rangeslider=dict(visible=True))) 
   
    # Plot
    if st.session_state.fig_2:    
        st.plotly_chart(figure_or_data = st.session_state.fig_2, use_container_width = True)
    
    #   Actualización del modelo
    if 'update_clicked' not in st.session_state:
        st.session_state.update_clicked = False

    if st.button(label = "Actualizar", key = "submit3", type = "primary"):
        st.session_state.update_clicked = True
        st.markdown("""
        ### La actualización conlleva:
        - Descarga de los indicadores económicos y del valor del 'Financial_Sector_Close' 
        - Descarga de las últimas noticias financieras
        - Análisis de sentimiento mediante el modelo 'model_glove_lstm'
        - Incorporación de estos datos al Dataframe inicial
        """)
        st.markdown("""
        Este proceso puede tardar unos minutos.
        
        - Nota 1: Las API-Keys de las distintas fuentes de consulta tienen 
            una duración determinada y podría ser necesaria su actualización manual.
        - Nota 2: Los datos se mostrarán a continuación.                    
        """)

    if st.session_state.update_clicked:    
        if st.button(label = "Continuar", key = "submit4"):
            if max_date < datetime.today():
                last_date = max_date.strftime('%Y-%m-%d')
                mensaje,df_trading_out = actualizar_mod(last_date)
                st.write(mensaje)                  
                with st.expander(label = "DataFrame actualizado", expanded = False):
                    st.dataframe(df_trading_out)
                    st.markdown(body = download_file(df = df_trading_out), unsafe_allow_html = True)
            else:
                st.write("A fecha actual los datos ya están actualizados")
        if st.button(label="Volver", key="submit5"):
            st.session_state.update_clicked = False
        
        


    # # DataFrame
    # with st.expander(label = "DataFrame", expanded = False):
    #     st.dataframe(datos_)
    #     st.markdown(body = download_file(df = datos_), unsafe_allow_html = True)




