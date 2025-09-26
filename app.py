import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

import google.generativeai as genai


import os
from dotenv import load_dotenv

load_dotenv()  # This loads variables from .env into environment

api_key = os.getenv("GOOGLE_API_KEY")
# print(api_key)
genai.configure(api_key=) 
model = genai.GenerativeModel("gemini-2.0-flash-lite")

model_ann = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer Churn Prediction')


def create_summary(customer_data, prediction_text):
    # Prepare the prompt for Gemini with relevant user data and prediction
    prompt = (
        "Here is a bank customer with the following information:\n"
        f"{customer_data}\n\n"
        f"Prediction: {prediction_text}\n"
        "Generate a detailed summary explaining the customer's profile and the churn prediction in simple terms."
    )
    # Call Gemini to generate the summary
    response = model.generate_content(prompt)
    return response.text



geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict churn
prediction = model_ann.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')



if st.button("Get Prediction and Summary"):
    # Predict churn
    prediction = model_ann.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    if prediction_proba > 0.5:
        result_text = "The customer is likely to churn."
    else:
        result_text = "The customer is not likely to churn."

    # Prepare customer data for Gemini
    customer_data = input_data.to_dict(orient="records")[0]

    summary = create_summary(customer_data, result_text)

    st.write("### Prediction")
    st.write(f"Churn Probability: {prediction_proba:.2f}")
    st.write(result_text)

    st.write("### AI-Generated Summary")
    st.write(summary)
    st.download_button(
        label="Download Summary",
        data=summary,
        file_name="customer_summary.txt",
        mime="text/plain"
    )
