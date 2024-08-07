import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tensorflow.keras.models import load_model

# Load pre-trained model and scaler
model = load_model('cnn_model.keras')
scaler = joblib.load('scaler.pkl')

# Function to predict new data
def predict_new_data(new_data):
    new_data_df = pd.DataFrame([new_data])
    new_data_df = new_data_df.rename(columns={'ed': 'education_level'})

    imputer = IterativeImputer(max_iter=10, random_state=0)
    new_data_df[['income', 'age']] = imputer.fit_transform(new_data_df[['income', 'age']])

    new_data_df['total_debt'] = new_data_df['creddebt'] + new_data_df['othdebt']
    new_data_df['debt_to_income_ratio'] = new_data_df['total_debt'] / new_data_df['income']
    new_data_df['employ_to_age_ratio'] = new_data_df['employ'] / new_data_df['age']

    columns_to_transform = ['age', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'total_debt', 'debt_to_income_ratio', 'employ_to_age_ratio']
    for col in columns_to_transform:
        transformed, _ = stats.yeojohnson(new_data_df[col])
        new_data_df[col] = np.clip(transformed, new_data_df[col].quantile(0.01), new_data_df[col].quantile(0.99))

    new_data_df[['age', 'education_level', 'employ', 'address']] = new_data_df[['age', 'education_level', 'employ', 'address']].astype(int)

    X_new = scaler.transform(new_data_df)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

    predictions_probability = model.predict(X_new)
    predictions_binary = (predictions_probability > 0.5).astype("int32")
    return predictions_binary[0][0]

# Streamlit app
def main():
    st.title("Loan Default Predictor")

    # Input fields for customer data
    age = st.number_input("Age", min_value=0)
    ed = st.number_input("Education Level(Graduate(0),High School(1), Postgraduate(2), Undergraduate(3))", min_value=0)
    employ = st.number_input("Employment Duration(Number of years the applicant has been employed.)", min_value=0)
    address = st.number_input("Address Duration(Number of years the applicant has lived at their current address.)", min_value=0)
    income = st.number_input("Income(Applicant's annual income in thousands of dollars.)", min_value=0.0)
    debtinc = st.number_input("Debt-to-Income Ratio(Ratio of the applicant's total monthly debt payments to their monthly gross income.)", min_value=0.0)
    creddebt = st.number_input("Credit Debt(Amount of credit card debt the applicant has.)", min_value=0.0)
    othdebt = st.number_input("Other Debt(Amount of other debts the applicant has.)", min_value=0.0)

    new_data = {
        'age': age, 'ed': ed, 'employ': employ, 'address': address,
        'income': income, 'debtinc': debtinc, 'creddebt': creddebt,
        'othdebt': othdebt
    }

    if st.button("Predict"):
        with st.spinner('Predicting...'):
            prediction = predict_new_data(new_data)
            st.success(f"Prediction: {'Default' if prediction == 1 else 'No Default'}")

if __name__ == "__main__":
    main()
