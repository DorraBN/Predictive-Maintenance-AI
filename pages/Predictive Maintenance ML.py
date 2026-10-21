import streamlit as st
import joblib
import pandas as pd
import os

from preprocessing import load_data, x_reg

st.set_page_config(
    page_title="Predictive Maintenance",
    layout="wide"
)

st.title("Predictive Maintenance - Logistic Regression")

# Load trained model
MODEL_PATH = "models/model_reg.joblib"

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run train_models.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Load dataset
@st.cache_data
def get_data():
    return load_data()

data = get_data()

# Metrics
st.subheader("Model Performance")

if os.path.exists("models/metrics_reg.joblib"):
    metrics = joblib.load("models/metrics_reg.joblib")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Recall", f"{metrics['recall']:.3f}")
    col3.metric("F1 Score", f"{metrics['f1']:.3f}")

    conf_matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual No Failure", "Actual Failure"],
        columns=["Predicted No Failure", "Predicted Failure"]
    )

    st.subheader("Confusion Matrix")
    st.dataframe(conf_matrix)
else:
    st.info("Metrics not found. Add the metrics-saving code in train_models.py.")



st.subheader("Enter machine sensor values")

air_temp = st.number_input(
    "Air temperature [K]", min_value=0.0, value=300.0
)

process_temp = st.number_input(
    "Process temperature [K]", min_value=0.0, value=310.0
)

rot_temp = st.number_input(
    "Rotational speed [rpm]", min_value=0, value=1500
)

tor_temp = st.number_input(
    "Torque [Nm]", min_value=0.0, value=40.0
)

tool_temp = st.number_input(
    "Tool wear [min]", min_value=0, value=100
)

machine_type = st.selectbox(
    "Machine type",
    ["H", "L", "M"]
)

h_temp = int(machine_type == "H")
l_temp = int(machine_type == "L")
m_temp = int(machine_type == "M")

if st.button("Predict machine failure"):

    # Keep the same feature names and order as training
    input_data = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rot_temp,
        "Torque [Nm]": tor_temp,
        "Tool wear [min]": tool_temp,
        "Type_H": h_temp,
        "Type_L": l_temp,
        "Type_M": m_temp
    }])

    # Reorder according to training features
    input_data = input_data[x_reg.columns]

    prediction = int(model.predict(input_data)[0])

    st.write("Result:", prediction)

    if prediction == 1:
        st.error("The machine is predicted to FAIL.")
    else:
        st.success("The machine is predicted NOT TO FAIL.")
