from preprocessing import load_data, x_reg, y_reg_encoded

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import joblib
import os

# Load data
df = load_data()

x_train, x_test, y_train, y_test = train_test_split(
    x_reg,
    y_reg_encoded,
    test_size=0.2,
    random_state=42
)

# Create models directory
os.makedirs("models", exist_ok=True)

# ---------------------------
# Logistic Regression
# ---------------------------

model_reg = LogisticRegression(max_iter=1000)
model_reg.fit(x_train, y_train)

y_predict_reg = model_reg.predict(x_test)

recallscore = recall_score(
    y_test, y_predict_reg, zero_division=0
)

accuracy = accuracy_score(y_test, y_predict_reg)

f1score = f1_score(
    y_test, y_predict_reg, zero_division=0
)

confusionmatrix = confusion_matrix(y_test, y_predict_reg)

classificationreport = classification_report(
    y_test, y_predict_reg, zero_division=0
)
joblib.dump({
    "accuracy": accuracy,
    "recall": recallscore,
    "f1": f1score,
    "confusion_matrix": confusionmatrix
}, "models/metrics_reg.joblib")
joblib.dump(model_reg, "models/model_reg.joblib")

# ---------------------------
# Deep Learning
# ---------------------------

model_dl = Sequential([
    Dense(32, activation="relu", input_shape=(x_train.shape[1],)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_dl.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model_dl.fit(
    x_train,
    y_train,
    epochs=30,
    batch_size=10,
    verbose=1
)

loss, dl_accuracy = model_dl.evaluate(
    x_test, y_test, verbose=0
)

print("DL Loss:", loss)
print("DL Accuracy:", dl_accuracy)

y_predict_d = (
    model_dl.predict(x_test, verbose=0) >= 0.5
).astype(int)

print(
    classification_report(
        y_test, y_predict_d, zero_division=0
    )
)


model_dl.save_weights("models/model_dl.weights.h5")

print("Models saved successfully!")