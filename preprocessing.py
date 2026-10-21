import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def load_data():
    return pd.read_csv("dataset.csv")


df = load_data()
def write_data(df):
    print("data description:",df.describe)
    print("missing values :",df.isnull().sum())
    print("duplicated values :",df.duplicated().sum())
    print("cat features :",df.select_dtypes(include="object").columns.tolist())
    print("numerical features :",df.select_dtypes(include="number").columns.tolist())

# Encode Type into Type_H, Type_L, Type_M
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

type_encoded = encoder.fit_transform(df[["Type"]])

type_df = pd.DataFrame(
    type_encoded,
    columns=encoder.get_feature_names_out(["Type"]),
    index=df.index
)

# Features
x_reg = df.drop(columns=[
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "Type"
])

x_reg = pd.concat([x_reg, type_df], axis=1)

# Target
y_reg_encoded = df["Machine failure"].astype(int).values

print("Features:", x_reg.columns.tolist())
print("Number of features:", x_reg.shape[1])
