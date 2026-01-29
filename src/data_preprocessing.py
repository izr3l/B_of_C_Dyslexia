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
  
    # Generic cleaning: drop duplicates
    df = df.drop_duplicates()
    
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
        if df['Dyslexia'].dtype == 'object':
            df['Dyslexia'] = df['Dyslexia'].map({'Yes': 1, 'No': 0})

    for col in df.columns:
        if 'Accuracy' in col or 'Missrate' in col:
            df[col] = df[col].apply(lambda x: x / 1000.0 if x > 1.0 else x)
            
    return df

def preprocess_data(df, target_column='Dyslexia'):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
        
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_numeric = X.select_dtypes(include=[np.number])
    
    # Handle any remaining NaNs
    X_numeric = X_numeric.fillna(X_numeric.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    # CRITICAL FIX: explicit index assignment to valid misalignment with y
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns, index=X_numeric.index)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
    
    # improved balancing using SMOTE (Synthetic Minority Over-sampling Technique)
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    except ImportError:
        # Fallback to manual oversampling if imblearn is somehow missing (though we installed it)
        train_df = pd.concat([X_train, y_train], axis=1)
        df_majority = train_df[train_df[target_column] == 0]
        df_minority = train_df[train_df[target_column] == 1]
        
        if len(df_minority) > 0:
            df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=42)
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
            X_train = df_balanced.drop(columns=[target_column])
            y_train = df_balanced[target_column]
    
    return X_train, X_test, y_train, y_test, scaler, X_numeric.columns
