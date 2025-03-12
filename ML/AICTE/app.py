import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the pipeline
with open("diabetes_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

def user_input_features():
    st.sidebar.header("Enter Patient Details")
    pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=300, value=100)
    blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
    skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=1000, value=80)
    bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
    diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
    
    # Create a DataFrame with the input data
    data = {
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    }
    df = pd.DataFrame(data)
    
    # Perform feature engineering (same as in p3.ipynb)
    def set_insulin(row):
        if 16 <= row["Insulin"] <= 166:
            return "Normal"
        else:
            return "Abnormal"
    
    df['NewInsulinScore'] = df.apply(set_insulin, axis=1)
    
    # Categorize BMI
    NewBMI_categories = ["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"]
    df['NewBMI'] = "Normal"  # Default value
    df.loc[df["BMI"] < 18.5, "NewBMI"] = "Underweight"
    df.loc[(df["BMI"] >= 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = "Normal"
    df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = "Overweight"
    df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = "Obesity 1"
    df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = "Obesity 2"
    df.loc[df["BMI"] > 39.9, "NewBMI"] = "Obesity 3"
    
    # Convert 'NewBMI' to categorical
    df['NewBMI'] = pd.Categorical(df['NewBMI'], categories=NewBMI_categories)
    
    # Categorize Glucose
    NewGlucose_categories = ["Low", "Normal", "Overweight", "Secret", "High"]
    df["NewGlucose"] = "Normal"  # Default value
    df.loc[df["Glucose"] <= 70, "NewGlucose"] = "Low"
    df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = "Normal"
    df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = "Overweight"
    df.loc[df["Glucose"] > 126, "NewGlucose"] = "High"
    
    # Convert 'NewGlucose' to categorical
    df['NewGlucose'] = pd.Categorical(df['NewGlucose'], categories=NewGlucose_categories)
    
    # One-hot encoding with all possible categories
    df = pd.get_dummies(df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)
    
    # Add missing columns (if any)
    expected_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
        'NewBMI_Normal', 'NewBMI_Overweight', 'NewBMI_Obesity 1', 'NewBMI_Obesity 2', 'NewBMI_Obesity 3',
        'NewInsulinScore_Normal', 'NewGlucose_Normal', 'NewGlucose_Overweight', 'NewGlucose_Secret', 'NewGlucose_High'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with default value 0
    
    # Ensure the columns are in the correct order
    df = df[expected_columns]
    
    return df

# Streamlit app title

st.title("Diabetes Prediction App")
st.write("Enter the required details in the sidebar and get the prediction result.")

# Get user input and perform feature engineering
input_df = user_input_features()

# Debug: show input data as built
st.subheader("Input Data")
st.write(input_df)

# Removed reindex call causing AttributeError
# input_df = input_df.reindex(columns=pipeline.final_columns, fill_value=0)

# Sidebar option to compare with p3.ipynb output
compare_output = st.sidebar.checkbox("Compare with p3.ipynb Output")

if st.button("Predict"):
    prediction = pipeline.predict(input_df)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.subheader(f"Prediction: {result}")
    if prediction == 1:
        st.warning("The model predicts that the patient may have diabetes. Please consult a doctor.")
    else:
        st.success("The model predicts that the patient is not diabetic.")

    # Compare with p3.ipynb prediction if selected
    if compare_output:
        p3_pred = pipeline.predict(input_df)[0]  # simulate p3.ipynb prediction
        result_p3 = "Diabetic" if p3_pred == 1 else "Non-Diabetic"
        st.subheader(f"p3.ipynb Prediction: {result_p3}")

# Debugging: Print pipeline details
st.subheader("Pipeline Details")
st.write(pipeline.named_steps['model'])