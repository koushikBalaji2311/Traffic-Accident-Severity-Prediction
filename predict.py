import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

def preprocess_input(user_input, label_encoders, scaler, numeric_cols):
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical values
    for column in label_encoders:
        if column in input_df:
            input_df[column] = label_encoders[column].transform(input_df[column])

    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    return input_df

def provide_recommendations(prediction):
    recommendations = {
        'Major Crash': (
            "Recommendations:\n"
            "- Enhance road safety infrastructure.\n"
            "- Conduct regular road safety audits.\n"
            "- Stricter enforcement of speed limits."
        ),
        'Minor Crash': (
            "Recommendations:\n"
            "- Improve driver awareness through campaigns.\n"
            "- Install better signage for hazardous areas.\n"
            "- Encourage seatbelt use."
        ),
        'Fatal Crash': (
            "Recommendations:\n"
            "- Target speeding and distracted driving.\n"
            "- Invest in crash prevention tech.\n"
            "- Strengthen emergency response systems."
        )
    }
    return recommendations.get(prediction, "No recommendations available.")

def predict_severity(user_input):
    model = lgb.Booster(model_file="models/lightgbm_model.txt")
    label_encoders = joblib.load("models/label_encoders.pkl")
    
    numeric_cols = ["vehicle_speed", "crash_time", "age", "number_of_lanes", "lane_width", "speed_limit"]
    scaler = joblib.load("models/scaler.pkl")
    
    processed_input = preprocess_input(user_input, label_encoders, scaler, numeric_cols)
    
    prediction = model.predict(processed_input)
    predicted_class = np.argmax(prediction)
    class_name = {0: "Major Crash", 1: "Minor Crash", 2: "Fatal Crash"}[predicted_class]
    
    print(f"Predicted Severity: {class_name}")
    print(provide_recommendations(class_name))

if __name__ == "__main__":
    user_input = {
        'vehicle_speed': 107,
        'crash_time': 11,
        'age': 27,
        'gender': 'Male',
        'vehicle_type': 'Heavy Vehicle',
        'number_of_lanes': 2,
        'lane_width': 3.48,
        'road_type': 'Urban',
        'alcohol_consumption': 'Yes',
        'crash_type': 'Rear-end',
        'seatbelt_usage': 'No',
        'speed_limit': 30,
        'road_surface_condition': 'Icy'
    }
    predict_severity(user_input)
