from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import joblib

N = 10  # Change for no. of records to predict in test_dataset

# Load the saved model
model = joblib.load('price_range_predictor.pkl')

test_data = pd.read_csv("test_dataset.csv")

test_data_sample = test_data.drop('id', axis=1).iloc[:N, :]  

# Make predictions on the test data sample
predictions = model.predict(test_data_sample)

# Print the predicted price ranges
print(f"Predicted Price Ranges for the First {N} Records:")
print(predictions)