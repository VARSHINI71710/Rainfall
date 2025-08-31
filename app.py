import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

with open("rainfall_rf_model.pkl", "rb") as file:
    model = pickle.load(file)


df=pd.read_csv("random_4000_rows.csv")

df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})


X = df.drop(columns=["RainTomorrow", "Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm"])
y = df["RainTomorrow"]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

st.title("Rainfall Prediction App 🌧️")
st.write("Predict whether it will rain tomorrow based on weather features")
st.subheader("Enter Weather Details")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter {col}", value=float(X[col].mean()))

user_df = pd.DataFrame([input_data])


if st.button("Predict"):
    prediction = rf.predict(user_df)[0]
    if prediction == 1:
        st.success("🌧️ Yes, it will rain tomorrow!")
    else:
        st.info("☀️ No, it will not rain tomorrow.")