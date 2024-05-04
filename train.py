import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Without ground truth labels (price_range) in the test_dataset.csv, I cannot directly 
# calculate evaluation metrics like confusion matrix and classification report for that dataset.
# Because of that I will split the train dataset 80/20 and use 20% for testing and evaluation.

data = pd.read_csv("train_dataset.csv")

# print(data.isnull().sum()) # check if there are empty fields

# print(data.dtypes) # check the data types, split between int and float

features = data.drop('price_range', axis=1) 
y = data['price_range']

import matplotlib.pyplot as plt

# Iterate through each feature and create a scatter plot with 'price_range' to get insights
# Do scatter plot for every feature against price_range 

feature_y = 'price_range'  

for feature_x in features:
    plt.scatter(data[feature_x], data[feature_y])
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.title(f'Scatter Plot: {feature_x} vs Price Range')
    plt.show()
    plt.clf()  

# I did the scatter plot for oberservation and the only feature that had a proportional relationship with price was ram.
# Thus it's safe to conclude that the price is influenced by a combination of features

# I choose Random Forest Classifier because it combines multiple decision trees, each potentially capturing different features' influence on price.
# It can also handle numerical and categorical features well, which is the case here

# Split data into training and validation sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier object
model = RandomForestClassifier(n_estimators=100)  # test n_estimators and found 100 to be the best

# Trained model
model.fit(X_train, y_train)

# Save the trained model using joblib
joblib.dump(model, 'price_range_predictor.pkl')  

################ VALIDATION TESTING  ################

# Make predictions on the validation set
predictions = model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, average='weighted')  
recall = recall_score(y_test, predictions, average='weighted')  
f1 = f1_score(y_test, predictions, average='weighted')  

# Confusion matrix and classification report
confusion_matrix_result = confusion_matrix(y_test, predictions)
classification_report_result = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

print("\nConfusion Matrix:\n", confusion_matrix_result)

print("\nClassification Report:\n", classification_report_result)

############# EXAMPLE OF RESULTS #############

# Accuracy: 0.86
# Precision: 0.8668598751040067
# Recall: 0.86
# F1 Score: 0.8612603925585265

# Confusion Matrix:
#  [[100   5   0   0]
#  [  8  73  10   0]
#  [  0   8  78   6]
#  [  0   0  19  93]]

# Classification Report:
#                precision    recall  f1-score   support 

#            0       0.93      0.95      0.94       105  
#            1       0.85      0.80      0.82        91  
#            2       0.73      0.85      0.78        92  
#            3       0.94      0.83      0.88       112  

#     accuracy                           0.86       400  
#    macro avg       0.86      0.86      0.86       400  
# weighted avg       0.87      0.86      0.86       400  