import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


### 1.- Datos del Sector Financiero

def get_fin(last_date):

    # Función para obtener datos de un índice de mercado o ETF usando la API de Yahoo Finance
    def get_market_data(symbol, start_date, end_date):
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            return data
        except Exception as e:
            print(f"Error getting data for {symbol} from Yahoo Finance: {e}")
            return None
        
    # Función para obtener VIX
    def get_vix_historical_data(start_date):
        # Símbolo del índice VIX en Yahoo Finance
        vix_ticker = "^VIX"
        # Definir las fechas de inicio y fin
        start_date = start_date
        end_date = datetime.today().strftime('%Y-%m-%d')
        # Obtener datos del VIX desde Yahoo Finance
        vix_data = yf.download(vix_ticker, start=start_date, end=end_date)
        return vix_data    

    # Función para obtener historical_volatility
    def get_historical_volatility(symbol, start_date, end_date, window_size=30):
        try:
            # Descargar datos históricos desde Yahoo Finance
            data = yf.download(symbol, start=start_date, end=end_date)
            # Calcular los rendimientos logarítmicos
            data['Log Returns'] = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
            # Calcular la volatilidad histórica
            data['Historical Volatility'] = data['Log Returns'].rolling(window=window_size).std() * np.sqrt(252) 
            # Volatilidad anualizada, es 252 porque es el numero de días que la bolsa está abierta
            data['Historical Volatility'] = data['Historical Volatility'] * 100  # Convertir a porcentaje
            data.reset_index(inplace=True)
            data = data[['Date', 'Adj Close', 'Historical Volatility']].dropna()
            return data        
        except Exception as e:
            print(f"Error getting data for {symbol} from Yahoo Finance: {e}")
            return None
        
    # Funciones para la obtención de alfa y beta.    
    def get_historical_prices(symbol, start_date, end_date):
        data = yf.download(symbol, start=start_date, end=end_date)
        return data['Adj Close']

    def calculate_daily_beta(stock_prices, market_prices, window_size=30):
        # Usamos 30 días para calcular la beta, lo que nos otorga fluctuaciones a más corto plazo.
        # Si quisiésemos enfocarnos más en el largo plazo, habría que aumentarlo.
        beta_values = []
        # Iterar 
        for i in range(len(stock_prices) - window_size + 1):
            stock_prices_window = stock_prices.iloc[i:i+window_size]
            market_prices_window = market_prices.iloc[i:i+window_size]            
            stock_returns = stock_prices_window.pct_change().dropna()
            market_returns = market_prices_window.pct_change().dropna()
            
            # Calcula beta como la covarianza de los rendimientos de la acción con los del mercado dividida 
            # por la varianza de los rendimientos del mercado.
            covariance = stock_returns.cov(market_returns)
            market_variance = market_returns.var()            
            beta = covariance / market_variance
            beta_values.append(beta)        
        return pd.Series(beta_values, index=stock_prices.index[window_size-1:])
    
    def calculate_daily_alpha(stock_prices, market_prices, risk_free_rate, window_size=30):
        alpha_values = []
        beta_values = calculate_daily_beta(stock_prices, market_prices, window_size)        
        # Iterar
        for i in range(len(stock_prices) - window_size + 1):
            stock_prices_window = stock_prices.iloc[i:i+window_size]
            market_prices_window = market_prices.iloc[i:i+window_size]            
            stock_returns = stock_prices_window.pct_change().dropna()
            market_returns = market_prices_window.pct_change().dropna()            
            beta = beta_values.iloc[i]  # Obtener la beta para el lapso de tiempo establecido            
            # Calculate market return (average daily return of market)
            market_return = market_returns.mean()            
            # Calcular el rendimiento esperado utilizando CAPM
            expected_return = risk_free_rate + beta * (market_return - risk_free_rate)            
            # Rendimiento real de la acción
            actual_return = stock_returns.mean()            
            # Calcular alfa
            alpha = actual_return - expected_return
            alpha_values.append(alpha)        
        return pd.Series(alpha_values, index=stock_prices.index[window_size-1:])
    
    def get_tresure_performance_10_years(start_date, end_date):
        # Símbolo del rendimiento del Tesoro a 10 años en Yahoo Finance
        tesoro_ticker = "^TNX"
        # Obtener datos históricos del rendimiento del Tesoro a 10 años
        treasure_performance = yf.download(tesoro_ticker, start=start_date, end=end_date)
        return treasure_performance['Adj Close']
    
    def get_treasury_yield(symbol, start_date, end_date):
        # Descargar datos históricos para el rendimiento del Tesoro a 10 años
        data = yf.download(symbol, start=start_date, end=end_date)
        return data['Adj Close']

    def format(df):
        if df is not None: 
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Date'}, inplace=True)         
            df['Date'] = pd.to_datetime(df['Date'])
        return df


    start_date = last_date
    end_date = datetime.today().strftime("%Y-%m-%d")

    # 1.1.- Obtener datos del Sector Financiero desde Yahoo Finance
    financial_sector_symbol = "XLF"
    financial_sector_data = get_market_data(financial_sector_symbol, start_date, end_date)
    if financial_sector_data is not None: 
        financial_sector_data.drop(columns=['Adj Close'], inplace=True)
    financial_sector_data = format(financial_sector_data)

    # 1.2.- Obtener datos históricos del VIX
    vix_data = get_vix_historical_data(start_date)
    if vix_data is not None: 
        vix_data.drop(columns=['Volume'], inplace=True)
    vix_data = format(vix_data)

    # 1.3.- Obtener datos de volatilidad histórica
    symbol = "XLF" 
    window_size = 30  # Lapso de tiempo para el cálculo de volatilidad
    new_start_date = datetime.strptime(start_date, '%Y-%m-%d')-timedelta(days=1.5*window_size)
    new_start_date = new_start_date.strftime("%Y-%m-%d")
    historical_volatility_data = get_historical_volatility(symbol, new_start_date, end_date, window_size)
    historical_volatility_data['Date'] = pd.to_datetime(historical_volatility_data['Date'])

    # 1.4.- Obtención de alfa y beta.
        # Obtener rendimientos históricos
    treasury_yield_symbol = "^TNX"
    treasury_yield = get_treasury_yield(treasury_yield_symbol, new_start_date, end_date)
    if treasury_yield is not None:
        average_risk_free_rate = treasury_yield.mean() / 100  # Convert percentage to decimal
    xlf_symbol = 'XLF'  # Financial Select Sector SPDR Fund
    gspc_symbol = '^GSPC'  # S&P 500 Index

        # Obtener precios históricos
    xlf_prices = get_historical_prices(xlf_symbol, new_start_date, end_date)
    gspc_prices = get_historical_prices(gspc_symbol, new_start_date, end_date)

        # Obtener el rendimiento histórico del Tesoro a 10 años y calcular la tasa libre de riesgo promedio
    treasure_performance = get_tresure_performance_10_years(new_start_date, end_date)
    if treasure_performance is not None:
            risk_free_rate = treasure_performance.mean() / 100  # Convertir porcentaje a decimal

        # Calcular alfa y beta
    alpha_values = calculate_daily_alpha(xlf_prices, gspc_prices, risk_free_rate)
    beta_values = calculate_daily_beta(xlf_prices, gspc_prices)

    results = pd.DataFrame({
        'Date': alpha_values.index,
        'Alpha': alpha_values.values,
        'Beta': beta_values.values  })
    results['Date'] = pd.to_datetime(results['Date'])
        
    # Combinar DataFrames basados en la columna 'Date'
    if financial_sector_data is not None and vix_data is not None:  
        merged_data = pd.merge(financial_sector_data[['Date', 'Open', 'Close', 'Volume']],
                            vix_data[['Date', 'Open', 'Close']],
                            on='Date', how='inner')
    if merged_data is not None and historical_volatility_data is not None:  
        merged_data = pd.merge(merged_data,
                            historical_volatility_data[['Date', 'Historical Volatility']],
                            on='Date', how='inner')
    if merged_data is not None and results is not None:  
        merged_data = pd.merge(merged_data,
                            results[['Date', 'Alpha', 'Beta']],
                            on='Date', how='inner')
        
    # Combinar DataFrames basados en la columna 'Date'
    merged_data.columns = [
        'Date', 
        'Financial_Sector_Open', 
        'Financial_Sector_Close', 
        'Volume', 
        'VIX_Open', 
        'VIX_Close', 
        'Historical_Volatility', 
        'Alpha', 
        'Beta'
    ]

    print ('get_financial success!')    
    return merged_data   