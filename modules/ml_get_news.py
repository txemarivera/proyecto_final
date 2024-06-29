import nltk
import os
import requests
import numpy as np
import pandas as pd

from datetime import datetime
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam

from modules.ml_func import *

### 2.- Datos del Noticias Financieras

def get_news(last_date):
    try:
        load_dotenv()
    except: pass
    api_token = os.environ['api_token']
    if api_token:
        print("api_token, OK!!!!!!!!!!!")
    from_date = last_date
    to_date = datetime.today().strftime('%Y-%m-%d')
    url_template = 'https://eodhd.com/api/news?s=AAPL.US&offset={}&limit=1000&api_token={}&fmt=json&from={}&to={}'

    # Función para obtener noticias financieras
    def fetch_news(from_date, to_date, api_token):
        
        all_news = []
        offset = 0
        while True:
            url = url_template.format(offset, api_token, from_date, to_date)
            response = requests.get(url)
            if response.status_code != 200:
                break
            data = response.json()
            if not data:
                break
            all_news.extend(data)
            data=[]
            offset += 1000

        if all_news:
            df_news = pd.DataFrame(all_news)
            # Filtrar por fecha si la columna 'date' está presente
            if 'date' in df_news.columns:
                df_news['date'] = pd.to_datetime(df_news['date'])
            
        return df_news
        
    # Trata el df obtenido para su posterior uso en el modelo lstm_glove de análisis de sentimiento
    def tratar_noticias(df_news):
        try:
            df=df_news[['date','title']].copy()
            df.rename(columns={'title': 'Sentence'}, inplace=True)
            df=df[~df['Sentence'].isna()]
            df = df.drop_duplicates("Sentence")
            return df
        except KeyError as e:
            print(f"Error: {e}")
            return pd.DataFrame()
    
    # Funciones de limpieza del texto
    def generar_stopwords(X):
        X = df['Sentence'].apply(limpiar_texto).values
        count_vectorizer = CountVectorizer(max_features = 8000)
        count_vectorizer.fit_transform(X)
        vocabulario_ordenado = sorted(count_vectorizer.vocabulary_.items(), key = lambda x : x[0], reverse=False)
        STOPWORDS = nltk.corpus.stopwords.words("english")
        lista_stopwords = [ item[0] for item in vocabulario_ordenado if item[0] > 'zz' or item[0] < 'aa']
        STOPWORDS = set(STOPWORDS).union(set(lista_stopwords))
        return STOPWORDS
    
    def limpiar_texto(texto):
        texto = texto.lower()
        texto_limpio = ""
        for s in texto:
            if s.isalnum() or s.isspace():
                texto_limpio += s
        return texto_limpio
    
    def eliminar_stopwords(texto):
        tokens = nltk.word_tokenize(text = texto)
        tokens = [token for token in tokens if token not in STOPWORDS]
        return " ".join(tokens) 
    
    def preproceso(texto):
        texto_clean = limpiar_texto(texto)
        tokens = eliminar_stopwords(texto_clean)
        return texto_clean
    
    # Función de vectorización del texto
    def vectorizer(X):
        word_tokenizer = Tokenizer()
        word_tokenizer.fit_on_texts(X)
        # vocab_length = len(word_tokenizer.word_index) + 1
        texts = X
        def embed(corpus): 
            return word_tokenizer.texts_to_sequences(corpus)

        longest_train = max(texts, key=lambda sentence: len(embed(sentence)))
        length_long_sentence = len(embed(longest_train))
        padded_sentences = pad_sequences(
            embed(texts), 
            length_long_sentence, 
            padding='post'
        )
        return padded_sentences

    # 1.- Descargar noticias
    df_news = fetch_news(from_date, to_date, api_token)
    if df_news is not None: 
        # 2.- Formato y limpieza de noticias  
        df = tratar_noticias(df_news)

        # 3.- Preparación de texto para su análisis
            # definir stopwords
        STOPWORDS = generar_stopwords (df['Sentence'])
            # ejecutar preproceso
        df['Sentence_clean'] = df['Sentence'].apply(preproceso)
        df=df.reset_index(drop = True)
        X = df['Sentence_clean'].values
        
        # 4.- Vectorizar texto
        padded_sentences = vectorizer(X)

        # 5.- Cargar el modelo 
        model_glove_lstm = load_model('./models/model_glove_lstm.keras', compile=False)
        model_glove_lstm.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy']) 
        
        # 6.- Ejecutar el modelo
        y_pred = model_glove_lstm.predict(padded_sentences)
        ypred = np.argmax(y_pred, axis = 1)
        y_pred_one_hot = np.eye(y_pred.shape[1])[ypred]
        
        # 7.- Añadir resultados a df
            # Generar dataframe con los resultados
        df_predit = pd.DataFrame(y_pred_one_hot, columns=['Negative', 'Neutral', 'Positive'])
            # Reordenar las columnas para que coincidan con los datos empleados en el modelo de model_trading
        df_predit = df_predit[['Negative', 'Positive', 'Neutral']]
            # unir con df
        df_ = pd.concat([df,df_predit],axis=1)   

        # 8.- Sumamos el número de noticias de cada tipo por día
        df_['date'] = pd.to_datetime(df_['date'], format='mixed')
        df_['date'] = df_['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df_fin = df_.groupby('date')[['date','Negative','Positive', 'Neutral']].agg({'Negative': 'sum', 'Positive': 'sum', 'Neutral': 'sum'}).reset_index()
    
    print (df_news)
    print ('get_news success!')
    
    return df_fin
