import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import streamlit as st
data = pd.read_csv("cancer_data.csv")
data = data.dropna() 
st.write(f"Loaded data with shape: {data.shape}")
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
y = y.map({'M': 1, 'B': 0})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
joblib.dump((model, scaler), "cancer_model.pkl")
def main():
    st.title("Cancer Diagnosis Prediction")
    st.write("Predict whether a tumor is B or M based on input features.")
    st.header("Input Features")
    input_features = {}
    for col in X.columns:
        input_features[col] = st.number_input(f"{col}", value=float(X[col].mean()), format="%.4f")
    input_data = pd.DataFrame([input_features])
    if st.button("Predict"):
        model, scaler = joblib.load("cancer_model.pkl")
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        st.subheader("Results")
        st.write(f"Prediction: **{'M' if prediction == 1 else 'B'}**")
        st.write(f"Probability of being M: {probabilities[1]:.2%}")
        st.write(f"Probability of being B: {probabilities[0]:.2%}")

if __name__ == "__main__":
    main()
