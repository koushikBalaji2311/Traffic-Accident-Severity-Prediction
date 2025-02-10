import pandas as pd
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_importance(model_path, feature_names):
    model = lgb.Booster(model_file=model_path)
    
    importance = model.feature_importance(importance_type='gain')
    feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10), palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance (Gain)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    return feature_importances

if __name__ == "__main__":
    feature_names = ["vehicle_speed", "crash_time", "age", "number_of_lanes", "lane_width", "speed_limit",
                     "gender", "vehicle_type", "road_type", "alcohol_consumption", "crash_type", "seatbelt_usage", "road_surface_condition"]
    plot_feature_importance("models/lightgbm_model.txt", feature_names)
