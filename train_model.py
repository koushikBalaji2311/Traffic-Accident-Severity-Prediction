import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model(data_path, model_path):
    data = pd.read_csv(data_path)
    
    # Encode categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    X = data.drop(columns=['crash_severity'])
    y = data['crash_severity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'max_depth': 6,
        'num_leaves': 31,
        'lambda_l1': 0.2,
        'lambda_l2': 0.2,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'random_state': 42
    }

    model_lgb = lgb.train(params, train_data, num_boost_round=500, valid_sets=[train_data, test_data],
                          callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=10)])

    model_lgb.save_model(model_path)
    joblib.dump(label_encoders, "models/label_encoders.pkl")
    print(f"Model trained and saved to {model_path}")


train_model("data/processed_data.csv", "models/lightgbm_model.txt")
