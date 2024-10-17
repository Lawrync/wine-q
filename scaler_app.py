import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the saved Random Forest model and scaler
random_forest_model = joblib.load('best_random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to make predictions
def make_prediction(features):
    # Scale the input features using the pre-trained scaler
    features_scaled = scaler.transform(np.array(features).reshape(1, -1))
    prediction = random_forest_model.predict(features_scaled)
    return prediction[0]

# Main function to run the app
def main():
    # Page title
    st.title("Wine Quality Prediction")

    # Sidebar with user inputs
    st.sidebar.title("Enter Wine Features")

    # Input fields for features (based on the provided columns)
    alcohol = st.sidebar.slider("Alcohol", min_value=0.0, max_value=20.0, step=0.1)       
    chlorides = st.sidebar.slider("Chlorides", min_value=0.0, max_value=0.2, step=0.001)  
    citric_acid = st.sidebar.slider("Citric Acid", min_value=0.0, max_value=1.0, step=0.01)
    density = st.sidebar.slider("Density", min_value=0.98, max_value=1.05, step=0.001)    
    fixed_acidity = st.sidebar.slider("Fixed Acidity", min_value=0.0, max_value=20.0, step=0.1)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", min_value=0, max_value=100, step=1)
    pH = st.sidebar.slider("pH", min_value=2.5, max_value=4.0, step=0.01)
    residual_sugar = st.sidebar.slider("Residual Sugar", min_value=0.0, max_value=15.0, step=0.1)
    sulphates = st.sidebar.slider("Sulphates", min_value=0.0, max_value=2.0, step=0.01)   
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", min_value=0, max_value=300, step=1)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", min_value=0.0, max_value=1.5, step=0.01)

    # Collect user inputs in a list
    features = [
        alcohol,
        chlorides,
        citric_acid,
        density,
        fixed_acidity,
        free_sulfur_dioxide,
        pH,
        residual_sugar,
        sulphates,
        total_sulfur_dioxide,
        volatile_acidity
    ]

    # Make prediction
    prediction = make_prediction(features)

    # Display the result
    st.subheader("Prediction:")
    if prediction == 3:
        st.write("The wine is predicted to be of **quality 3** (very poor quality).")     
    elif prediction == 4:
        st.write("The wine is predicted to be of **quality 4** (poor quality).")
    elif prediction == 5:
        st.write("The wine is predicted to be of **quality 5** (low quality).")
    elif prediction == 6:
        st.write("The wine is predicted to be of **quality 6** (standard quality).")      
    elif prediction == 7:
        st.write("The wine is predicted to be of **quality 7** (high quality).")
    elif prediction == 8:
        st.write("The wine is predicted to be of **quality 8** (very high quality).")     
    elif prediction == 9:
        st.write("The wine is predicted to be of **quality 9** (excellent quality).")     
    else:
        st.write("The wine quality prediction is not within the expected range.")

    # Optionally: Display the predicted value directly
    st.write(f"Predicted Wine Quality: {prediction}")

# Run the app
if __name__ == "__main__":
    main()
   
   



