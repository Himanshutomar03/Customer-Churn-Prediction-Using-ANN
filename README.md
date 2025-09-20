Click here to view the deployed app https://customer-churn-prediction-using-ann-3pqbwzyn72flq6yptvxa2g.streamlit.app/



# Customer Churn Prediction Using ANN

This project predicts customer churn for a bank using an Artificial Neural Network (ANN) built with TensorFlow/Keras. It includes data preprocessing, model training, and deployment as both a Jupyter notebook and a Streamlit web app.

## Features

- **Data Preprocessing:** Cleans and encodes the raw dataset, including label encoding for gender and one-hot encoding for geography.
- **Model Training:** Trains an ANN to predict whether a customer will leave the bank (churn) using key features.
- **Reusable Preprocessing:** Saves encoders and scaler as pickle files for consistent preprocessing during prediction and deployment.
- **Local Prediction Notebook:** Allows you to test predictions on new data in a Jupyter notebook.
- **Streamlit Web App:** Interactive web interface for entering customer details and viewing churn predictions.

## How It Works

1. **Preprocessing & Training:**  
   - Load and clean the dataset.
   - Encode categorical features and scale numerical features.
   - Train an ANN model and save it along with preprocessing objects.

2. **Prediction:**  
   - Load the trained model, encoders, and scaler.
   - Preprocess new customer data to match the model’s expected input.
   - Predict churn probability and display results.

3. **Deployment:**  
   - Use the Streamlit app for interactive predictions.
   - All preprocessing steps are consistent with training for reliable results.

## Files

- 1.ipynb: Data preprocessing, encoding, model training, and saving objects.
- prediction.ipynb: Local notebook for making predictions on new data.
- app.py: Streamlit web app for user-friendly churn prediction.
- requirements.txt: List of required Python packages.

## Demo

Try the deployed app: [Live Demo](https://customer-churn-prediction-using-ann-3pqbwzyn72flq6yptvxa2g.streamlit.app/)

