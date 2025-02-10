# Accident Severity Prediction

## Project Overview
This repository contains a machine learning project designed to predict the severity of accidents based on various accident-related features. Using a dataset with details such as vehicle speed, driver demographics, road conditions, and crash type, the goal is to develop a predictive model that classifies accidents into different severity levels.

## Features
- **Data Preprocessing**: Handles missing values, standardizes numeric features, and encodes categorical variables.
- **Accident Severity Prediction**: Uses machine learning models to classify accidents as fatal, serious, or minor.
- **Feature Importance Analysis**: Identifies key factors contributing to accident severity.
- **Model Evaluation**: Assesses performance using accuracy, precision, recall, and other classification metrics.

## Technologies Used
- Python
- Pandas & NumPy
- Scikit-learn
- LightGBM
- Matplotlib & Seaborn
- Joblib (for model saving/loading)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Prepare Data**: Ensure that the dataset is available at the specified file path.
2. **Run the Notebook**: Execute the Jupyter Notebook to preprocess data, train models, and visualize results.
3. **Evaluate Performance**: Check classification metrics and feature importance plots.

## File Structure
- `Hackathon1.ipynb` - Main notebook for data processing, model training, and evaluation.
- `requirements.txt` - List of required Python packages.
- `models/` - Saved models for reuse.
- `data/` - Folder containing the dataset.

## Authors
 Koushik Balaji P
 Shree Pranav S
 Joel Ebenezer P

## License
This project is licensed under the MIT License.

