import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()
nltk.download('stopwords')
nltk.download('wordnet')

######################### Funciones auxiliares #########################
# Función para preprocesar el texto
def preprocess(text):
    # Pasamos a minúsculas
    text = text.lower()
    # Dejamos solo palabras
    text = re.sub(r'[^\w\s]|[\d]', ' ', text)
    # Eliminar espacios adicionales
    text = re.sub('\s+', ' ', text).strip()
    # Lematizo
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split(' ')]
    # Filtrar las stopwords
    stop_words = stopwords.words('english')
    text = ' '.join([word for word in tokens if ((word not in stop_words) and (len(word)>=2))])
    return text

# Función para realizar los conteos de palabras
def word_count(str):
    # Create an empty dictionary named 'counts' to store word frequencies.
    counts = dict()
    # Split the input string 'str' into a list of words using spaces as separators and store it in the 'words' list.
    words = str.split()
    # Iterate through each word in the 'words' list.
    for word in words:
        # Check if the word is already in the 'counts' dictionary.
        if word in counts:
            # If the word is already in the dictionary, increment its frequency by 1.
            counts[word] += 1
        else:
            # If the word is not in the dictionary, add it to the dictionary with a frequency of 1.
            counts[word] = 1
    # Return the 'counts' dictionary, which contains word frequencies.
    return counts

######################### Preparamos los datos para streamlit #########################
# Cargar datos, eliminamos duplicados, aplicamos el prerpocesado y guardamos esa versión
df = pd.read_csv("data_news_sentiment.csv")
df.drop_duplicates("Sentence", inplace=True)
df['Sentence_prepro'] = df.Sentence.progress_apply(preprocess)
df.to_csv('data_news_sentiment_prepro.csv', index=False)

# Cargamos la versión preprocesada y eliminamos nulos
df = pd.read_csv("data_news_sentiment_prepro.csv")
df = df.dropna()

# Para cada categoría generamos los conteos de palabras
for cat in df.Sentiment.unique():
    dict_conts = word_count(' '.join(df[df.Sentiment==cat].Sentence_prepro.values))
    df_conts = pd.DataFrame(data=dict_conts.items(), columns=['Word', 'Count']).sort_values(by='Count', ascending=False)
    df_conts.to_csv(f'word_counts_{cat}.csv', index=False)