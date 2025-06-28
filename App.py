import streamlit as st
import joblib
import numpy as np

# Load model and feature names
model = joblib.load("life_expectancy_model.pkl")
features = [
    'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
    'Hepatitis B', 'Measles ', 'BMI', 'under-five deaths ', 'Polio',
    'Total expenditure', 'Diphtheria ', 'HIV/AIDS', 'GDP', 'Population',
    'thinness  1-19 years', 'thinness 5-9 years', 'Income composition of resources',
    'Schooling'
]

st.title("🌍 Life Expectancy Predictor")

st.markdown("Enter the values for each input feature below:")

user_input = []
for feature in features:
    value = st.number_input(f"**{feature}**", step=0.1)
    user_input.append(value)

if st.button("Predict Life Expectancy"):
    input_array = np.array(user_input).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted Life Expectancy: **{prediction:.2f} years**")
