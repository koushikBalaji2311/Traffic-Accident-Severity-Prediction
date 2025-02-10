import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Drop target column if already exists
    data = data.drop(columns=['crash_severity'], errors='ignore')

    # Convert numeric columns and fill missing values
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].median())

    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data


processed_data = load_and_preprocess_data("give your data path")
processed_data.to_csv("data/processed_data.csv", index=False)
print("Data preprocessing completed and saved.")
