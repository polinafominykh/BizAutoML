# core/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df, target_column):
    df = df.copy()

    # 1. Заполнение пропусков
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Missing')
        else:
            df[col] = df[col].fillna(df[col].median())

    # 2. Кодирование категориальных признаков
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # 3. Масштабирование
    scaler = StandardScaler()
    feature_columns = [col for col in df.columns if col != target_column]
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # 4. Разделение признаков и целевой переменной
    X = df[feature_columns]
    y = df[target_column]

    return X, y, scaler, label_encoders
