import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder , LabelEncoder

model = tf.keras.models.load_model('churn_model.h5')

        
with open('onehot_geography.pkl', 'rb') as f:
    ohe = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


st.title('Customer Churn Prediction')

geography = st.selectbox('Select Geography', options=ohe.categories_[0])
gender = st.selectbox('Select Gender',options=encoder.classes_)
age = st.slider('Enter Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Enter Balance', min_value=0)


credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)

tenure = st.slider('Enter Tenure', min_value=0, max_value=10, value=5)
num_of_products = st.number_input('Number of Products',min_value=1)

has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


input_data = {
    'CreditScore': credit_score,
    'Gender': encoder.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary


}
input_df = pd.DataFrame([input_data])

geo_df = pd.DataFrame({'Geography': [geography]})
geo_encoded = ohe.transform(geo_df)
geo_df = pd.DataFrame(geo_encoded.toarray(), columns=ohe.get_feature_names_out(['Geography']))
input_full = pd.concat([input_df.reset_index(drop=True), geo_df], axis=1)



input_scaled = scaler.transform(input_full)
prediction = model.predict(input_scaled)
churn = prediction[0][0] > 0.5

if churn:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')
st.write(f'Churn probability: {prediction[0][0]:.2f}')