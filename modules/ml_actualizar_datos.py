import numpy as np
import pandas as pd



### 3.- Actualizar el dataframe de entrada de datos del modelo de trading 

def actualizar_datos(df_fin, df_news):

    un_dia = pd.Timedelta(days=1)
    max_dias = 4

    df_fin['date'] = pd.to_datetime(df_fin['Date'], format='ISO8601').dt.tz_localize(None)
    df_news['date'] = pd.to_datetime(df_news['date'], format='ISO8601').dt.tz_localize(None)           
            
    df_fin[['Negative','Positive', 'Neutral']]= 0
    for date in df_news['date'].unique():
        if date in df_fin['date'].values:
            df_fin.loc[df_fin['date'] == date, ['Negative','Positive', 'Neutral']] = df_news.loc[df_news['date'] == date, ['Negative','Positive', 'Neutral']].values[0]
        # el número máximo de días sin datos de bolsa es 4
        else:
            for i in range(1,max_dias+1):
                prev_date = date - i * un_dia
                if prev_date in df_fin['date'].values:
                    df_fin.loc[df_fin['date'] == prev_date, ['Negative','Positive', 'Neutral']] += df_news.loc[df_news['date'] == date, ['Negative','Positive', 'Neutral']].values[0]
                    break
                    
                if prev_date < df_fin['date'].min():
                    break        
    df_trading = df_fin.drop(['Date', 'Financial_Sector_Open', 'VIX_Open'], axis=1)                    
   
    # Unimos los datos con los anteriores
    df_trading_input = pd.read_csv("./data/inputs/df_trading_input.csv")
    df_trading_out = pd.concat([df_trading_input,df_trading], axis=0).reset_index(drop=True)
    df_trading_out['date'] = pd.to_datetime(df_trading_out['date'], format='ISO8601').dt.tz_localize(None) 

    df_trading_out.to_csv('./data/outputs/df_trading_out.csv',index=False)

    print ('actualizar_datos success!')
    
    return df_trading_out

    
