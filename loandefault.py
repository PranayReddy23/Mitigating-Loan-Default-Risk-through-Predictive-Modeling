import streamlit as st
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Function to predict new data
def predict_new_data(model, scaler, new_data):
    try:
        new_data_df = pd.DataFrame([new_data])
        new_data_df = new_data_df.rename(columns={'ed': 'education_level'})

        # Feature engineering
        new_data_df['total_debt'] = new_data_df['creddebt'] + new_data_df['othdebt']
        
        # Avoid division by zero for debt_to_income_ratio
        new_data_df['debt_to_income_ratio'] = new_data_df['total_debt'] / np.where(new_data_df['income'] == 0, 1, new_data_df['income'])
        
        # Avoid division by zero for employ_to_age_ratio
        new_data_df['employ_to_age_ratio'] = new_data_df['employ'] / np.where(new_data_df['age'] == 0, 1, new_data_df['age'])

        # Ensure columns are int32 for compatibility with the model
        new_data_df[['age', 'education_level', 'employ', 'address']] = new_data_df[['age', 'education_level', 'employ', 'address']].astype('int32')

        # Check for any infinity or NaN values after feature engineering
        if np.any(np.isinf(new_data_df)) or np.any(np.isnan(new_data_df)):
            raise ValueError("Feature engineering resulted in NaN or infinite values.")

        X_new = scaler.transform(new_data_df)
        X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)  # Reshape for CNN

        predictions_probability = model.predict(X_new)
        predictions_binary = (predictions_probability > 0.5).astype("int32")
        return predictions_binary[0][0]

    except Exception as e:
        st.error(f"Error predicting new data: {e}")
        return None  # Return None on error to signal to Streamlit


# Streamlit app
def main():
    st.title("Credit Default Prediction")

    # Load pre-trained model and scaler
    model = load_model('cnn_model.keras')  
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Input fields
    st.subheader("Enter Customer Information:")
    age = st.number_input("Age", min_value=0, value=35)
    ed = st.number_input("Education Level(High School(1), Undergraduate(3), Graduate(1), Postgraduate(2))", min_value=0, value=2)
    employ = st.number_input("Employment Duration (Number of years the applicant has been employed.)", min_value=0, value=5)
    address = st.number_input("Time at Current Address (Number of years the applicant has lived at their current address.)", min_value=0, value=2)
    income = st.number_input("Income (Applicant's annual income in thousands of dollars.)", min_value=0.0, value=50000.0)
    debtinc = st.number_input("Debt-to-Income Ratio (Ratio of the applicant's total monthly debt payments to their monthly gross income.)", min_value=0.0, value=15.0)
    creddebt = st.number_input("Credit Card Debt(Amount of credit card debt the applicant has.)", min_value=0.0, value=1000.0)
    othdebt = st.number_input("Other Debt(Amount of other debts the applicant has.)", min_value=0.0, value=2000.0)
    
    new_data = {
        'age': age, 'ed': ed, 'employ': employ, 'address': address,
        'income': income, 'debtinc': debtinc, 'creddebt': creddebt,
        'othdebt': othdebt
    }

    if st.button("Predict"):
        prediction = predict_new_data(model, scaler, new_data)
        if prediction is not None:
            st.success(f"Prediction: {'Default' if prediction == 1 else 'No Default'}")

if __name__ == "__main__":
    main()
