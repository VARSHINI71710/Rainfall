🌧️ Rainfall Prediction Project

App link - https://rainfall-predictor1.streamlit.app/

📌 Overview

This project predicts Rain Tomorrow (Yes/No) using machine learning models trained on historical weather data.
The model is saved as a .pkl file and deployed using Streamlit for interactive predictions.

📂 Dataset

The dataset contains weather-related features such as:

🌡️ Temperature

💧 Humidity

🌧️ Rainfall amount

🌬️ Wind Speed

☀️ Sunshine hours

🎯 Target Column: RainTomorrow (Yes / No)

⚙️ Steps Done
🧹 Data Preprocessing

Handled missing values

Encoded categorical columns

Normalized skewed features

📊 Data Visualization

Plotted correlation heatmap to identify feature importance

Visualized rainfall vs. non-rainfall distribution

Checked feature distributions before and after scaling

⚖️ Handling Imbalance

Applied SMOTE (Synthetic Minority Oversampling Technique)

Balanced dataset → improved model performance

🤖 Model Training

Used Random Forest Classifier with GridSearchCV for hyperparameter tuning

Achieved high accuracy and balanced metrics

📈 Evaluation

Generated Classification Report

✅ Accuracy

✅ Precision

✅ Recall

✅ F1-Score

Plotted Confusion Matrix

💾 Model Saving

Trained model saved as rainfall_rf_model.pkl

🌐 Streamlit App

Interactive web app built using Streamlit:

Users input weather details (Temp, Humidity, Rainfall, Wind, Sunshine, etc.)

Model predicts → "Will it rain tomorrow?"

✅ Example Output in Streamlit:

🌤️ Weather Prediction Result:
It will RAIN Tomorrow: YES 🌧️


or

🌤️ Weather Prediction Result:
No Rain Tomorrow ☀️

🛠️ Requirements

Install dependencies with:

pip install -r requirements.txt


requirements.txt

pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
xgboost
streamlit
ipykernel

🚀 How to Run

1️⃣ Clone the repo / copy project files

git clone <repo-link>
cd rainfall-prediction


2️⃣ Install requirements

pip install -r requirements.txt


3️⃣ Run the Streamlit app

streamlit run app.py

📊 Sample Results
🔎 Classification Report
Accuracy: 0.88
Precision: 0.85
Recall: 0.83
F1-Score: 0.84

📈 Confusion Matrix

✅ Correctly predicted most "No Rain" days

✅ Improved recall for "Rain" after applying SMOTE

✨ Final Outcome: A Rainfall Prediction App with robust preprocessing, balanced dataset, trained Random Forest model, and interactive Streamlit deployment 🚀
