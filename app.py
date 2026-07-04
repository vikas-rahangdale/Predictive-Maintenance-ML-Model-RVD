# supreme
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import  BaggingClassifier 



# Load the saved model and preprocessing objects
model = joblib.load("best_model_before_smote.pkl")
# Load the scaler
scaler = joblib.load("scaler.pkl")
# Load the feature names
feature_names = joblib.load("feature_names.pkl")

# streamlit app
st.title("Application for Predictive Maintenance of Machines")
st.image("Maint_Pred.png", width=500)
st.write("Enter the below details of machine to predict whether a machine is likely to experience a failure or not , so that appropriate maintenance actions can be taken in time.")

#Categorical input fields
type=st.selectbox("Type of Machine", ["L", "M", "H"])

#numeric input fields
airtemperaturek=st.number_input("Air temperature [K]", min_value=0.0, max_value=500.0, value=300.0)
processtemperaturek=st.number_input("Process temperature [K]", min_value=0.0, max_value=500.0, value=300.0)
rotationalspeedrpm=st.number_input("Rotational speed [rpm]", min_value=0.0, max_value=5000.0, value=1500.0)
torquenm=st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=50.0)
toolwearmin=st.number_input("Tool wear [min]", min_value=0.0, max_value=500.0, value=100.0)



# Predict button
if st.button("Predict"):
    # Create a DataFrame from the input features
    input_data = pd.DataFrame({
        "type": [type],
        "airtemperaturek": [airtemperaturek],
        "processtemperaturek": [processtemperaturek],
        "rotationalspeedrpm": [rotationalspeedrpm],
        "torquenm": [torquenm],
        "toolwearmin": [toolwearmin]
    })

    # Preprocess the input data (e.g., scaling)
   # input_data.columns = input_data.columns.str.strip().str.replace(" ", "").str.replace("[", "").str.replace("]", "").str.replace("/", "").str.lower()
    
    input_data= pd.get_dummies(input_data, columns=['type'], drop_first=True, dtype=int)
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    input_data_scaled = scaler.transform(input_data)





    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Display the prediction result
    if prediction[0] == 1:
        st.success("Machine is likely to experience a failure soon. Maintenance is required to prevent unexpected downtime and ensure optimal performance.")
    else:
        st.success("Machine is not likely to experience a failure soon. No immediate maintenance is required, but regular monitoring and maintenance should still be performed to ensure continued performance.")







