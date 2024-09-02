import streamlit as st
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import keras
import os
import warnings

# Suppress warnings and disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore', category=FutureWarning)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Data preprocessing function
def preprocess_data(df):
    df = df.rename(columns={'ed': 'education_level'})
    df['default'] = pd.to_numeric(df['default'], errors='coerce').fillna(0).astype(int)
    
    imputer = IterativeImputer(max_iter=10, random_state=0)
    df[['income', 'age']] = imputer.fit_transform(df[['income', 'age']])
    
    education_mode = df['education_level'].mode()[0]
    df['education_level'] = df['education_level'].fillna(education_mode)
    
    df['total_debt'] = df['creddebt'] + df['othdebt']
    df['debt_to_income_ratio'] = df['total_debt'] / df['income']
    df['employ_to_age_ratio'] = df['employ'] / df['age']
    
    columns_to_transform = ['age', 'employ', 'address', 'income', 'debtinc', 'creddebt', 'othdebt', 'total_debt', 'debt_to_income_ratio', 'employ_to_age_ratio']
    for col in columns_to_transform:
        transformed, _ = stats.yeojohnson(df[col])
        df[col] = np.clip(transformed, df[col].quantile(0.01), df[col].quantile(0.99))
    
    df[['age', 'education_level', 'employ', 'address']] = df[['age', 'education_level', 'employ', 'address']].astype(int)
    
    X = df.drop('default', axis=1)
    y = df['default']
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    X_train = X_train.reshape(X_train.shape[0], X.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    return X_train, X_test, y_train, y_test, scaler

# Function to train the model
def train_model(X_train, X_test, y_train, y_test, scaler):
    best_params = {
        'filters': [384, 256, 128],
        'kernel_sizes': [5, 7, 3],
        'dropout_rates': [0.23, 0.49],
        'learning_rate': 0.00018,
        'batch_size': 1024
    }
    
    model = Sequential()
    model.add(Conv1D(filters=best_params['filters'][0], kernel_size=best_params['kernel_sizes'][0], activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=best_params['filters'][1], kernel_size=best_params['kernel_sizes'][1], activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=best_params['filters'][2], kernel_size=best_params['kernel_sizes'][2], activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(best_params['dropout_rates'][0]))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(best_params['dropout_rates'][1]))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=best_params['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train, y_train, epochs=25, batch_size=best_params['batch_size'],
        validation_data=(X_test, y_test)
    )
    
    model.save('cnn_model.keras')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype("int32")
    classification_rep = classification_report(y_test, y_pred_binary, output_dict=True)
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    
    return model, history, classification_rep, confusion_mat

# Function to predict new data
def predict_new_data(model, scaler, new_data):
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
    st.title("ML Model Trainer and Predictor")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())
        
        if st.button("Preprocess Data"):
            X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.scaler = scaler
            st.success("Data preprocessed successfully!")
    
    if "X_train" in st.session_state:
        if st.button("Train Model"):
            with st.spinner('Training the model...'):
                try:
                    model, history, classification_rep, confusion_mat = train_model(st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test, st.session_state.scaler)
                    st.session_state.model = model
                    st.success("Model trained and saved successfully!")
                    st.subheader("Classification Report")
                    st.write(pd.DataFrame(classification_rep).transpose())
                    st.subheader("Confusion Matrix")
                    st.write(confusion_mat)
                except Exception as e:
                    st.error(f"Error training the model: {e}")
    
    if "model" in st.session_state:
        st.subheader("Predict New Data")
        age = st.number_input("Age", min_value=0)
        ed = st.number_input("Education Level", min_value=0)
        employ = st.number_input("Employment Duration", min_value=0)
        address = st.number_input("Address Duration", min_value=0)
        income = st.number_input("Income", min_value=0.0)
        debtinc = st.number_input("Debt-to-Income Ratio", min_value=0.0)
        creddebt = st.number_input("Credit Debt", min_value=0.0)
        othdebt = st.number_input("Other Debt", min_value=0.0)
        
        new_data = {
            'age': age, 'ed': ed, 'employ': employ, 'address': address,
            'income': income, 'debtinc': debtinc, 'creddebt': creddebt,
            'othdebt': othdebt
        }
        
        if st.button("Predict"):
            with st.spinner('Predicting...'):
                try:
                    model = st.session_state.model
                    with open('scaler.pkl', 'rb') as f:
                        scaler = pickle.load(f)
                    prediction = predict_new_data(model, scaler, new_data)
                    st.success(f"Prediction: {'Default' if prediction == 1 else 'No Default'}")
                except Exception as e:
                    st.error(f"Error predicting new data: {e}")

if __name__ == "__main__":
    main()
