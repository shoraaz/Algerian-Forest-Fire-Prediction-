import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the Ridge Regressor model and Standard Scaler from pickle files
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Title of the app
st.title("Forest Fire Prediction App")

# Sidebar for user input
st.sidebar.header("Input Parameters")

def user_input_features():
    Temperature = st.sidebar.number_input("Temperature", min_value=-10.0, max_value=50.0, value=15.0)
    RH = st.sidebar.number_input("Relative Humidity (RH)", min_value=0.0, max_value=100.0, value=50.0)
    Ws = st.sidebar.number_input("Wind Speed (Ws)", min_value=0.0, max_value=100.0, value=5.0)
    Rain = st.sidebar.number_input("Rain", min_value=0.0, max_value=50.0, value=0.0)
    FFMC = st.sidebar.number_input("FFMC", min_value=0.0, max_value=100.0, value=80.0)
    DMC = st.sidebar.number_input("DMC", min_value=0.0, max_value=100.0, value=40.0)
    ISI = st.sidebar.number_input("ISI", min_value=0.0, max_value=100.0, value=10.0)
    Classes = st.sidebar.number_input("Classes", min_value=0.0, max_value=1.0, value=0.5)
    Region = st.sidebar.number_input("Region", min_value=0.0, max_value=10.0, value=1.0)

    data = {
        'Temperature': Temperature,
        'RH': RH,
        'Ws': Ws,
        'Rain': Rain,
        'FFMC': FFMC,
        'DMC': DMC,
        'ISI': ISI,
        'Classes': Classes,
        'Region': Region
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Standardize the input data
scaled_data = standard_scaler.transform(input_df)

# Predict using the Ridge Regressor model
if st.button("Predict"):
    prediction = ridge_model.predict(scaled_data)
    st.success(f"The predicted output is: {prediction[0]:.2f}")
