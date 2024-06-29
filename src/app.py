import sys
import os

# Agregar la ruta del directorio raíz del proyecto al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Ahora puedes importar tus módulos normalmente
from modules.ml_func import *


import plotly.express as px
import streamlit as st
from PIL import Image

from modules.ml_func import *
from about import about_app
from eda_1 import eda_1_app
from eda_2 import eda_2_app
from ml import ml_app



def main():
    
    st.set_page_config(**PAGE_CONFIG)
    

    menu = ["Main App", "Exploratory NEWS Data Analysis","Exploratory FINANCIAL Data Analysis", "Machine Learning Model", "About"]

    choice = st.sidebar.selectbox(label = "Menu", options = menu, index = 0)
    
    if choice == "Main App":

        st.title(body = "ULTIMATE TRADING MODEL :chart_with_upwards_trend:")

        st.write("Welcome to the **ULTIMATE TRADING MODEL Website**.")

        st.write("""
        This project consists of predicting stock values based on:
        - The temporal evolution of the value itself.
        - The main market parameters such as volatility, market risk, etc.
        - Sentiment analysis of financial news.
                 """)

        st.write("""Go to `Exploratory Data Analysis` section to know more about the data that we used to build
                    the Machine Learning models.""")
        
        st.write("""Go to the `Machine Learning Model` section, you can either use the sliders in the sidebar.""")
        st.write("""Go to the `About` section for more information about the app and the people who built it.""")

            
        image = Image.open("./data/inputs/portada.png")

        st.image(image            = image,
                use_column_width = True)
 

    if choice == "Exploratory NEWS Data Analysis":
        eda_1_app()

    elif choice == "Exploratory FINANCIAL Data Analysis":
        eda_2_app()

    elif choice == "Machine Learning Model":
        ml_app()

    elif choice == "About":
        about_app()


if __name__ == "__main__":
    main()