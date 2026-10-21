import streamlit as st
import pandas as pd
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

st.set_page_config(
    page_title="Predictive Maintenance DL",
    layout="wide"
)

st.title("Predictive Maintenance DL")

MODEL_PATH = "models/model_dl.weights.h5"

# Load model weights
if not os.path.exists(MODEL_PATH):
    st.error(
        "Model weights not found. "
        "Run train_models.py first."
    )
    st.stop()

@st.cache_resource
def load_dl_model():
    model = Sequential([
        Dense(32, activation="relu", input_shape=(8,)),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.load_weights(MODEL_PATH)
    return model

model = load_dl_model()


st.subheader("Enter machine sensor values")

air_temp = st.number_input(
    "Air temperature [K]",
    min_value=0.0,
    value=300.0,
    key="air_temp"
)

process_temp = st.number_input(
    "Process temperature [K]",
    min_value=0.0,
    value=310.0,
    key="process_temp"
)

rot_temp = st.number_input(
    "Rotational speed [rpm]",
    min_value=0,
    value=1500,
    key="rot_temp"
)

tor_temp = st.number_input(
    "Torque [Nm]",
    min_value=0.0,
    value=40.0,
    key="tor_temp"
)

tool_temp = st.number_input(
    "Tool wear [min]",
    min_value=0,
    value=100,
    key="tool_temp"
)

machine_type = st.selectbox(
    "Machine type",
    ["H", "L", "M"]
)

h_temp = int(machine_type == "H")
l_temp = int(machine_type == "L")
m_temp = int(machine_type == "M")

if st.button("Predict machine failure"):

    input_data = pd.DataFrame(
        [[
            air_temp,
            process_temp,
            rot_temp,
            tor_temp,
            tool_temp,
            h_temp,
            l_temp,
            m_temp
        ]],
        columns=[
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type_H",
            "Type_L",
            "Type_M"
        ]
    )

    prediction = float(
        model.predict(input_data, verbose=0)[0][0]
    )

    st.metric(
        "Predicted failure probability",
        f"{prediction * 100:.2f}%"
    )

    if prediction >= 0.5:
        st.error("The machine is predicted to FAIL.")
    else:
        st.success("The machine is predicted NOT TO FAIL.")