import gradio as gr
import joblib
import pandas as pd
import os

# Load model
model = joblib.load("polynomial_regression (1).pkl")
poly = joblib.load("polynomial_features (1).pkl")


def predict_bill(ac_unit, fan_unit):

    input_data = pd.DataFrame({
        "AC_Units": [ac_unit],
        "Fan_Units": [fan_unit]
    })

    input_data_poly = poly.transform(input_data)

    prediction = model.predict(input_data_poly)[0]

    return f"Predicted Electricity Bill: {prediction:.2f}"


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

    outputs=gr.Textbox(label="Prediction"),

    title="Electricity Bill Prediction",

    description="Predict Electricity Bill based on AC Units and Fan Units."
)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
