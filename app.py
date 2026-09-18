import streamlit as st
import joblib
import pandas as pd
model = joblib.load("polynomial_regression(1).pkl")
poly = joblib.load("polynomial_features(1).pkl")
st.title("Electricity Bill Prediction")
ac_unit = st.number_input(
    "Enter AC Units",
    min_value=1.0,
    max_value=150.0,
    value=30.0
)
fan_unit = st.number_input(
    "Enter Fan Units",
    min_value=1.0,
    max_value=150.0,
    value=30.0
)
if st.button("Predict"):

    new_data = pd.DataFrame({
        "AC_Units": [ac_unit]
        "Fan_Units":[fan_unit]

    })

    new_data_poly = poly.transform(new_data)

    prediction = model.predict(new_data_poly)

    st.success(f"Predicted Electricity Bill: {prediction[0]:.2f}")
