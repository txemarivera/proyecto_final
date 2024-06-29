import streamlit as st


def about_app():
    # Título de la aplicación
    st.title("Trading en Streamlit")

    # Sección principal de la aplicación
    st.write("Bienvenido a nuestra aplicación.")

    # Añadir una sección 'About'
    # st.sidebar.title("About")
    # st.sidebar.info("""
    
    st.subheader("About")
    st.write("""
    **Propósito de la Aplicación**

    Esta aplicación de trading, desarrollada en Streamlit, está diseñada para ayudar a las personas usuarias a analizar y gestionar riesgos financieros asociados con diversas inversiones. 
    Utiliza datos de mercado en tiempo real y permite a las personas usuarias evaluar parámetros clave de volatilidad, riesgo de mercado, liquidez y más. 
    La aplicación es útil tanto para traders principiantes como para profesionales que buscan tomar decisiones informadas basadas en datos.

    **Información del Desarrollador o del Equipo**

    La aplicación ha sido desarrollada por un equipo compuesto por:

    - **Marco Bonaccorsi**: Jr Data Science. [LinkedIn](https://www.linkedin.com/in/marco-bonaccorsi-gonz%C3%A1lez/?originalSubdomain=es
                )
    - **María Hermo**: Jr Data Science. [LinkedIn](https://www.linkedin.com/in/mariahermorodriguez/?originalSubdomain=es)
    - **Txema Rivera**: Jr Data Science. [LinkedIn](https://www.linkedin.com/in/txema-rivera-sar/)

    **Tecnologías Utilizadas**

    - **Streamlit**: Framework de Python para la creación de aplicaciones web interactivas.
    - **Python**: Lenguaje de programación principal utilizado para el desarrollo.
    - **Bibliotecas de Datos**: Pandas, Tensorflow, NumPy, Plotly para la manipulación y visualización de datos.
    - **APIs Financieras**: Integración con plataformas como Yahoo Finance, API de frankfurter, API de Eodhd para la obtención de datos de mercado en tiempo real.

    **Instrucciones de Uso**

    - **Inicio**: Al abrir la aplicación, los usuarios pueden (lo completamos al final ya que no tengo claro que vamos poder filtrar).
    - **Parámetros**: Poderemos filtrar distintos parámetros de análisis deseados del EDA, indicadores tales como volatilidad histórica, índice VIX, beta, y otros indicadores relevantes.
    - **Resultados**: La aplicación mostrará gráficos interactivos y tablas con los datos analizados.
    
    **Opciones de mejora del proyecto** 
                
    - Entrenar el modelo con más datos.             
    - Actualizar el modelo con datos nuevos de forma automática
    - Emplear otras variables bursátiles
    - Análisis de sensibilidad: filtrar noticias por temas relacionados con el valor que se quiere predecir         
    - Para la predicción a 'N' días se mantienen constantes los parámetros que no son target. Se podrían realizar un modelo de predicción para esos parámetros y tomarlos como entradas en el modelo de trading.  
             
    **Créditos y Reconocimientos**

    Agradecemos a las siguientes personas y organizaciones que han contribuido o inspirado el desarrollo de esta aplicación:

    - Bloomberg y Yahoo Finance: Por proporcionar acceso a datos de mercado esenciales.
    - Plataformas de análisis de datos como TradingView y Google Finance.
    - Tutores y profesores de Hack a Boss y a los/las futuribles Data Science, nuestros compañeros/as por su la paciencia
    durante todo el proceso formativo.
    - Y nosotros mismos, Marco, Txema y María, que nos hemos apoyado un 100% en la realización de este proyecto.
    """)          

