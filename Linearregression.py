import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Import Linear Regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the data from the CSV file
file_path = "/content/drive/MyDrive/crop_yield.csv"
data = pd.read_csv(file_path)

# Print the first few rows to understand the structure of the dataset
print(data.head())

# Filter the data to only include rows where the Crop is "Rice"
rice_data = data[data['Crop'] == 'Rice']

# Convert categorical columns to numerical using LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column in the dataframe
for column in rice_data.select_dtypes(include=['object']).columns:
    rice_data[column] = label_encoder.fit_transform(rice_data[column].astype(str).str.strip())

# Define features (X) and target (y)
X = rice_data.drop('Yield', axis=1)  # Features (adjust column names if necessary)
y = rice_data['Yield']               # Target (yield)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on test data
y_pred = lr_model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE) and R-squared (R²)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Linear Regression Mean Squared Error (MSE): {mse:.4f}")
print(f"Linear Regression R-squared (R²): {r2:.4f}")

# Calculate accuracy of predictions based on actual yield
accuracy = 100 * (1 - (np.abs(y_test - y_pred) / y_test))
accuracy_df = pd.DataFrame({
    'Actual Yield': y_test,
    'Predicted Yield': y_pred,
    'Accuracy': accuracy
})

# Select top 10 predictions based on actual yield
top_10_predictions = accuracy_df.nlargest(10, 'Actual Yield')

print("Top 10 Yield Predictions:")
print(top_10_predictions)
