import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

from wordcloud import WordCloud

def eda_1_app():

    st.title('Análisis de Sentimientos de las Noticias')
    st.write("""
        Este código implementa una aplicación en Streamlit para realizar análisis de sentimientos en un 
        conjunto de noticias. Carga datos, realiza análisis exploratorio, visualiza distribuciones, filtra por
        sentimientos y procesa texto usando stemmer y lemmatizer. La finalidad es ofrecer una herramienta
        interactiva para analizar y visualizar sentimientos en noticias.
    """)

    df = pd.read_csv("./data/inputs/data_news_sentiment_prepro.csv")
    df.drop_duplicates("Sentence", inplace=True)
    df = df.dropna()

    # Pedir al usuario las categorías a analizar en el EDA
    categoria = st.multiselect(label='Filtra los sentimientos para visualizar los datos:', options=set(df.Sentiment.values))

    # Comprobar que el usuario haya seleccionado algo
    if len(categoria)>0:
        # Filtramgitos el df a petición del usuario
        df_eda = df[df.Sentiment.isin(categoria)]

        # 1. Mostramos las primeras filas del df a estudiar
        st.write(df_eda.head(3))

        # Total de noticias
        st.write(f'Total de noticias para las categorías seleccionadas: {len(df_eda)}.')

        # Realizamos el conteo de palabras
        df_eda['word_count'] = df_eda['Sentence'].apply(lambda x: len(str(x).split()))

        # 2. Ploteamos un histograma de conteos de palabras
        st.subheader('Distribución de la Longitud de Textos')
        st.write("""
            El histograma muestra la distribución de la longitud de los textos en el dataset. La mayoría de los
            textos tienen una longitud entre 30 y 70 caracteres, con un pico notable alrededor de los 40
            caracteres. La frecuencia disminuye gradualmente para textos más largos, siendo raros los textos que 
            superan los 150 caracteres. Esto indica que la mayoría de las entradas son relativamente cortas. Indica Bins a visualizar:
        """)

            # Número de bins
        bins = st.number_input(label='Bins', min_value=10, max_value=110, step=10)
            # Crear histograma con Plotly
        fig = px.histogram(df_eda, x='word_count', nbins=bins, title='Distribución de la Longitud de Textos',
                        labels={'word_count': 'Longitud del Texto'},
                        color_discrete_sequence=['#4682B4'])
        fig.update_layout(
            xaxis_title='Longitud del Texto',
            yaxis_title='Frecuencia',
            bargap=0.1,
            template='simple_white'
        )
            # Mostrar el histograma 
        st.plotly_chart(fig, use_container_width=True)

        # 3. Histograma de frecuencia de categorías
        st.subheader('Total de noticias de cada categoría')
            # Contar ocurrencias de cada etiqueta
        counts = df_eda['Sentiment'].value_counts().sort_values(ascending=False).reset_index()
        counts.columns = ['Sentiment', 'Count']
            # Crear el gráfico de barras con colores gradientes de azul
        colors = ['#0d47a1', '#1976d2', '#64b5f6']  
        # Crear el gráfico de barras
        fig = px.bar(
            counts,
            x='Sentiment',
            y='Count',
            color='Count',
            color_continuous_scale=colors,
            title='Sentimiento de las noticias',
            labels={'Sentiment': 'Etiqueta', 'Count': 'Cantidad'},
            text='Count'
        )
            # Ajustar el diseño del gráfico
        fig.update_layout(
            xaxis_title='Etiqueta',
            yaxis_title='Cantidad',
            coloraxis_showscale=False  # Ocultar la escala de colores
        )
            # Mostrar el gráfico en Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # 4. Longitudes según categoría
        st.subheader('Total de palabras para cada categoría')
        # Crear el gráfico de caja con Plotly
        fig = px.box(df_eda, x='Sentiment', y='word_count', color='Sentiment',
                    title='Boxplot de longitudes de texto por sentimiento',
                    labels={'Sentiment': 'Etiqueta', 'word_count': 'Longitud del Texto'},
                    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
        # Ajustar el diseño del gráfico
        fig.update_layout(
            xaxis_title='Etiqueta',
            yaxis_title='Longitud del Texto',
            template='simple_white'
        )
        # Mostrar el gráfico
        st.plotly_chart(fig, use_container_width=True)

        # 5. Estudio de la frecuencia de palabras
        st.subheader('Palabras más/menos frecuentes')
        st.write("""
            Filtra las palabas más frecuentes negativas y positivas.
        """)
        aparicion_min = st.number_input(label='Al menos la palabra debe aparecer...:', min_value=1, max_value=110, step=10)
        top_mas = st.number_input(label='Top_Más', min_value=10, max_value=110, step=10)
        top_menos = st.number_input(label='Top_Menos', min_value=10, max_value=110, step=10)
        fig, axs = plt.subplots(len(categoria), 2, figsize=(14, 5 * len(categoria)), squeeze=False) 
        for i, cat in enumerate(categoria):
            df_counts = pd.read_csv(f'./data/inputs/word_counts_{cat}.csv')
            df_counts = df_counts[df_counts.Count>=aparicion_min]
            df_mas = df_counts.head(top_mas)
            df_menos = df_counts.tail(top_menos)
            axs[i,0].set_title(f'Palabras más frecuentes de {cat}')
            df_mas.plot('Word', 'Count', kind='barh', color='#4682B4', ax=axs[i,0], yticks=[])
            axs[i,1].set_title(f'Palabras menos frecuentes de {cat}')
            df_menos.plot('Word', 'Count', kind='barh', color='#4682B4', ax=axs[i,1], yticks=[])
        st.pyplot(fig)

        # 6. WordCloud
        st.subheader('Nube de palabras')
        st.write("""
            Visualiza una nube de palabras con todo el df completo. ¿Cuántas quieres visualizar?
        """)
        max_words = st.number_input(label='Máximo de palabras', min_value=10, max_value=500, step=10)
        fig1, ax = plt.subplots()
        wordcloud = WordCloud(background_color='white', max_words=max_words)
        wordcloud.generate(' '.join(df.Sentence_prepro.values))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig1)