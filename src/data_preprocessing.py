import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """
    Loads the dataset from the CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None

def clean_data(df):
    """
    Cleans the dataset:
    - Handles missing values (if any)
    - Removes outliers (negative times)
    - Encodes categorical variables
    """
    # Remove rows with negative values in time-related columns if they exist
    # Assuming columns like 'Clicks', 'Hits', 'Score' are numeric.
    # We will filter out obvious bad data if known.
    
    # Generic cleaning: drop duplicates
    df = df.drop_duplicates()
    
    # Handle specific known issues (from plan: negative times)
    # Identifying time columns based on context or names if possible.
    # For now, we will just ensure non-negativity on numeric columns where appropriate.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
         df = df[df[col] >= 0]

    # Encode Gender if it exists and is categorical
    if 'Gender' in df.columns:
        if df['Gender'].dtype == 'object':
             le = LabelEncoder()
             df['Gender'] = le.fit_transform(df['Gender'])
    
    # Ensure Target 'Dyslexia' is properly formatted
    if 'Dyslexia' in df.columns:
        # If it's Yes/No, map to 1/0
        if df['Dyslexia'].dtype == 'object':
            df['Dyslexia'] = df['Dyslexia'].map({'Yes': 1, 'No': 0})
            
    return df

def preprocess_data(df, target_column='Dyslexia'):
    """
    Preprocesses the data for training:
    - Separates X and y
    - Scales features
    - Splits into train/test
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify non-numeric columns for OneHotEncoding if any remain?
    # For this dataset description, it seems mostly numeric + Gender.
    # We'll just select numeric types to be safe for scaling.
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Handle any remaining NaNs
    X_numeric = X_numeric.fillna(X_numeric.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, X_numeric.columns
