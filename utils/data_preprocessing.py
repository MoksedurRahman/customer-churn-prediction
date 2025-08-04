import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

def load_data(file_path):
    """Load and return raw data"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    return pd.read_csv(path)

def preprocess_data(df):
    """Clean and preprocess raw data"""
    # Convert TotalCharges to numeric (handling empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Convert target
    df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Drop ID column
    df.drop('customerID', axis=1, inplace=True)
    
    # Handle categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].nunique() == 2:
            df[col] = LabelEncoder().fit_transform(df[col])
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    # Scale numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    return df.drop('Churn', axis=1), df['Churn']