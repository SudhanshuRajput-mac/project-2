import streamlit as st
import pickle
import numpy as np

# Load the saved model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="ML Model Deployment", page_icon="🤖", layout="centered")

st.title("🤖 Machine Learning Model Deployment")
# st.write("This app uses a pre-trained model to make predictions.")

# Example: assume model takes 2 features as input
f1 = st.number_input("Enter The Age 👤 : ")
f2 = st.number_input("Enter Feature experience 🧑‍💻 : ")

# Predict button
if st.button("Predict"):
    input_data = np.array([[f1, f2]])  # Convert to 2D array
    prediction = model.predict(input_data)
    st.success(f"✅ Model Prediction: {prediction[0]}")
