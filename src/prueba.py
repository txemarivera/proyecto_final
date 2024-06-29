### Importar librerías
import numpy as np
import pandas as pd


import pickle



# Normalización
from sklearn.preprocessing import MinMaxScaler


# Métricas para Clasificación
from sklearn.metrics import confusion_matrix

from tensorflow.keras.layers import Input, Dense,  LSTM
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPool1D, Conv1D, Dropout, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import plotly.express as px

df_trading_input = pd.read_csv(r"..\data\inputs\df_trading_input.csv")

# Se toma el 80% de los datos para train. Se escala sobre ese 80%
datos = df_trading_input
len_train = int(0.80*len(datos))
datos = df_trading_input.drop(['date'], axis=1)


X = datos.drop('Financial_Sector_Close', axis=1)
y = datos['Financial_Sector_Close']

X_scaler = MinMaxScaler()
X_scaler.fit_transform(X[:len_train])
X_esc = X_scaler.transform(X)

y_scaler = MinMaxScaler()
y_ = np.array(y)
y_scaler.fit_transform(y_[:len_train].reshape(-1, 1))
y_esc = y_scaler.transform(y_.reshape(-1, 1))


# guardar scalers
with open('../models/y_scaler.pkl', 'wb') as f:
    pickle.dump(y_scaler, f) 
with open('../models/X_scaler.pkl', 'wb') as f:
    pickle.dump(X_scaler, f) 

scaler = MinMaxScaler()
scaler.fit_transform(datos[:len_train])
scaled_data = scaler.transform(datos)

datos = pd.DataFrame(scaled_data, columns=datos.columns)

with open('../models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)     

T = 3 # Segmentos
X = list()
y = list()

for t in range(len(datos) - T):
    
    # Toma valores de X de t en t con stride de 1
    x = datos[t : t + T]
    X.append(x)
    
    # Toma los valores de t en t
    y_ = datos.iloc[t + T, -1]
    y.append(y_)

# Transformamos a np.array y ajustamos las dimensiones
# Para entrar en el modelo debe de tener 3 dimensiones
X = np.array(X).reshape(-1, T, datos.shape[1] ) ##  ndim = datos.shape[1]

y = np.array(y).reshape(-1, 1, 1 ) 

print(f"X: {X.shape}\ty: {y.shape}")    



## Modelo
    
def trading():
    
    model = Sequential()

    model.add(Input(shape = (T, datos.shape[1]))) # T, 1) en lugar 1 sería ndim datos.shape[1]

    model.add(LSTM(units = 512, activation = "relu"))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1024, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 256, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(units = 64, activation = "relu")) 
    model.add(Dropout(0.2))
    model.add(Dense(units = 32, activation = "relu")) 
    model.add(Dropout(0.2))
    model.add(Dense(units = 16, activation = "relu")) 

    model.add(Dense(units = 1))

    model.compile(optimizer = "adam", loss = "mse")
    return model
    

model = trading()

model = trading()

earlystopping = EarlyStopping(
    min_delta=0.000001, 
    patience=100
)
checkpoint = ModelCheckpoint(
    '../models/model_trading.keras', 
    monitor = 'val_loss', 
    verbose = 1, 
    save_best_only = True
)
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.2, 
    verbose = 1, 
    patience = 5,                        
    min_lr = 0.001
)
history = model.fit(
    x = X[:len_train],
    y = y[:len_train],
    epochs = 1000,
    batch_size = 32,
    validation_data = (X[len_train:], y[len_train:]),
    verbose = 1
    ,callbacks = [checkpoint, earlystopping]
)