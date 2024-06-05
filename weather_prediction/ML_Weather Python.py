# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
data = pd.read_csv('model_training/weatherHistory.csv')

# Display the first few rows of the dataset
print(data.head())

# Display summary information about the dataset
print(data.info())
print(data.describe())

# Selecting features and target variable
features = data.drop(columns=['Temperature (C)'])  # Use the correct column name
target = data['Temperature (C)']  # Use the correct column name

# Drop non-numeric columns from features
features = features.drop(columns=['Formatted Date', 'Summary', 'Daily Summary'])

# Handle 'Snow' as a new category
if 'Snow' in features['Precip Type'].unique():
    # 'Snow' is already present, so no need to do anything
    print("Snow is already present as a category.")
else:
    # 'Snow' is not present, so add it as a new category
    features['Precip Type'].replace(np.nan, 'Snow', inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
features['Precip Type'] = label_encoder.fit_transform(features['Precip Type'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train_scaled)

# Make predictions on the testing set
y_test_pred = model.predict(X_test_scaled)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = train_mse ** 0.5
train_r2 = r2_score(y_train, y_train_pred)

test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training Mean Squared Error: {train_mse}")
print(f"Training Root Mean Squared Error: {train_rmse}")
print(f"Training R² Score: {train_r2}")

print()

print(f"Testing Mean Squared Error: {test_mse}")
print(f"Testing Root Mean Squared Error: {test_rmse}")
print(f"Testing R² Score: {test_r2}")

# Save the model and scaler
joblib.dump(model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
