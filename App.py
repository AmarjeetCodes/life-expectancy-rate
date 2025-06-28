
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Title
st.title("🌍 Life Expectancy Predictor")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Life Expectancy Data.csv")
    df.rename(columns = {
        'Life expectancy': 'Life_expectancy',
        'Adult Mortality': 'Adult_Mortality',
        'infant deaths': 'Infant_Deaths',
        'Hepatitis B': 'Hepatitis_B',
        'under-five deaths': 'Under_five_deaths',
        'BMI': 'BMI',
        'thinness 1-19 years': 'Thinness_10_19',
        'thinness 5-9 years': 'Thinness_5_9',
        'Income composition of resources': 'Income_Composition',
        'Total expenditure': 'Total_Expenditure'
    }, inplace=True)
    df.dropna(inplace=True)
    return df

df = load_data()

# Features and target
features = ['Alcohol', 'BMI', 'GDP', 'Schooling']
target = 'Life_expectancy'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model training
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Sidebar inputs
st.sidebar.header("Enter Health & Economic Indicators")
alcohol = st.sidebar.slider("Alcohol", 0.0, 20.0, 5.0)
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 20.0)
gdp = st.sidebar.slider("GDP", 0.0, 130000.0, 5000.0)
schooling = st.sidebar.slider("Schooling (years)", 0.0, 20.0, 12.0)

# Prediction
input_df = pd.DataFrame([[alcohol, bmi, gdp, schooling]], columns=features)
prediction = model.predict(input_df)[0]

# Display result
st.subheader("📈 Predicted Life Expectancy")
st.success(f"{prediction:.2f} years")
