import gradio as gr
import joblib
import pandas as pd

# Load model and polynomial features
model = joblib.load("polynomial_regression (1).pkl")
poly = joblib.load("polynomial_features (1).pkl")


# Prediction function
def predict_bill(ac_unit, fan_unit):

    new_data = pd.DataFrame({
        "AC_Units": [ac_unit],
        "Fan_Units": [fan_unit]
    })

    # Apply polynomial transformation
    new_data_poly = poly.transform(new_data)

    # Predict electricity bill
    prediction = model.predict(new_data_poly)

    return f"Predicted Electricity Bill: {prediction[0]:.2f}"


# Gradio interface
demo = gr.Interface(
    fn=predict_bill,
    inputs=[
        gr.Number(
            label="Enter AC Units",
            minimum=1,
            maximum=150,
            value=30
        ),
        gr.Number(
            label="Enter Fan Units",
            minimum=1,
            maximum=150,
            value=30
        )
    ],
    outputs=gr.Textbox(label="Electricity Bill"),
    title="Electricity Bill Prediction",
    description="Enter AC and Fan units to predict the electricity bill."
)

# Run the application
demo.launch()
