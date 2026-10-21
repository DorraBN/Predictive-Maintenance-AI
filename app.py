import streamlit as st
from preprocessing import write_data,load_data
import matplotlib.pyplot as plt
st.set_page_config(page_title="predictive maintenae",layout="wide")
st.title("Predictive Maintenance AI")
@st.cache_data
def get_data():
    return load_data()
data=get_data()
write_data(data)
st.write("dataset")
st.write(data.head())

numf=data["Machine failure"].value_counts()[1]
numnf=data["Machine failure"].value_counts()[0]
st.write("number of machines with failure:",numf)
st.write("number of machines with no failure:",numnf)
fig, ax = plt.subplots(figsize=(2, 2))
ax.pie(
    [numnf, numf],
    labels=["No Failure", "Failure"],
    autopct="%1.1f%%",
    startangle=90
)

ax.set_title("Machine Failure Distribution", fontsize=9)

st.pyplot(fig, use_container_width=False)


Predictive=st.Page(
    "pages/Predictive Maintenance ML.py"
)

st.subheader("Numerical Features Analysis")

st.line_chart(
    data,
    x="Product ID",
    y=[
        "Air temperature [K]",
        "Process temperature [K]"
    ]
)



type_counts = data['Type'].value_counts()

fig, ax = plt.subplots(figsize=(2, 2))

ax.pie(
    type_counts,
    labels=type_counts.index,
    autopct='%1.1f%%',
    startangle=90
)

ax.set_title("Distribution of Machine Types")

st.pyplot(fig)

faq=st.Page("pages/FAQ.py")
dl=st.Page(
    "pages/Predictive Maintenance DL.py"
)