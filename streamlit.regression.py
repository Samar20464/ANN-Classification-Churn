
import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model("regression_model.h5")

# Load the encoders and scaler
with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit application
st.title("Estimated Salary Prediction")
st.write("Enter the customer details to predict the estimated salary.")

# Input fields
geography = st.selectbox(
    "Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

credit_score = st.number_input(
    "Credit Score", min_value=0, max_value=1000, value=650
)

age = st.slider("Age", 18, 92, 35)
tenure = st.slider("Tenure", 0, 10, 5)
balance = st.number_input("Balance", min_value=0.0, value=0.0)
num_of_products = st.slider("Number of Products", 1, 4, 1)

has_credit_card = st.selectbox(
    "Has Credit Card", [0, 1]
)

is_active_member = st.selectbox(
    "Is Active Member", [0, 1]
)

exited = st.selectbox("Exited", [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "Exited": [exited]
})

# Encode geography
geo_encoded = onehot_encoder_geo.transform(
    [[geography]]
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
)

# Combine numerical and encoded categorical data
input_data = pd.concat(
    [input_data.reset_index(drop=True),
     geo_encoded_df.reset_index(drop=True)],
    axis=1
)

# Ensure the feature order matches the training data
if hasattr(scaler, "feature_names_in_"):
    input_data = input_data[scaler.feature_names_in_]

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict estimated salary
if st.button("Predict Salary"):
    prediction = model.predict(input_data_scaled, verbose=0)
    predicted_salary = float(prediction[0][0])

    st.success(
        f"Predicted Estimated Salary: ${predicted_salary:,.2f}"
    )
