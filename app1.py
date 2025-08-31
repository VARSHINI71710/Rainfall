import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# --- Load model ---
with open("rainfall_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- Page configuration ---
st.set_page_config(
    page_title="Rainfall Prediction App",
    page_icon="🌦️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #ff7f0e;
    }
    .stButton button {
        background-color: #1f77b4;
        color: white;
        font-size:16px;
        border-radius:10px;
        padding:10px 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.title("🌧️ Rainfall Prediction App")
st.write("Predict whether it will rain tomorrow based on weather features")
st.subheader("Enter Weather Details")

# --- Load dataset ---
df = pd.read_csv("random_4000_rows.csv")
df["RainTomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
df["RainToday"] = df["RainToday"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["RainTomorrow", "WindGustDir", "WindDir9am", "WindDir3pm"])
y = df["RainTomorrow"]

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# --- User input ---
input_data = {}
for col in X.columns:
    if col == "RainToday":
        input_data[col] = st.selectbox(
            f"Enter {col}", 
            options=["No", "Yes"]
        )
        # Convert to numeric
        input_data[col] = 1 if input_data[col] == "Yes" else 0
    else:
        input_data[col] = st.number_input(
            f"Enter {col}", 
            value=float(X[col].mean())
        )



user_df = pd.DataFrame([input_data])

# --- Prediction button ---
if st.button("Predict"):
    prediction = rf.predict(user_df)[0]
    if prediction == 1:
        st.success("🌧️ Yes, it will rain tomorrow!")
    else:
        st.info("☀️ No, it will not rain tomorrow.")

st.markdown('</div>', unsafe_allow_html=True)
