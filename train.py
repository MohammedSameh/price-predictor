import pandas as pd
import numpy as np
from sklearn import preprocessing

data = pd.read_csv("train_dataset.csv")

# print(data.isnull().sum()) # check if there are empty fields

# print(data.dtypes) # check the data types, split between int and float

features = data.drop('price_range', axis=1) 
y = data['price_range']

import matplotlib.pyplot as plt

# Iterate through each feature and create a scatter plot with 'price_range' to get insights
# Do scatter plot for every feature against price_range 

feature_y = 'price_range'  

# for feature_x in features:
#     plt.scatter(data[feature_x], data[feature_y])
#     plt.xlabel(feature_x)
#     plt.ylabel(feature_y)
#     plt.title(f'Scatter Plot: {feature_x} vs Price Range')
#     plt.show()
#     plt.clf()  

# I did the scatter plot for oberservation and the only feature that had a proportional relationship with price was ram.
# Thus it's safe to conclude that the price is influenced by a combination of features

# I choose Random Forest Classifier because it combines multiple decision trees, each potentially capturing different features' influence on price.
# It can also handle numerical and categorical features well, which is the case here

from sklearn.ensemble import RandomForestClassifier
import joblib

# Create a Random Forest Classifier object
model = RandomForestClassifier(n_estimators=100)  # test n_estimators and found 100 to be the best

# Trained model
model.fit(features, y)

# Save the trained model using joblib
joblib.dump(model, 'price_range_predictor.pkl')  


