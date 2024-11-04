import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the data from the CSV file
file_path = "/content/drive/MyDrive/crop_yield.csv"
data = pd.read_csv(file_path)

# Print the first few rows to understand the structure of the dataset
print("Original Dataset:")
print(data.head())

# Check unique values in the Crop column
print("Unique values in Crop column:", data['Crop'].unique())

# Filter the dataset to include only rows where the crop type is 'rice'
rice_data = data[data['Crop'].str.strip().str.lower() == 'rice']  # Ensure case insensitivity and strip spaces

# Print the filtered rows to confirm
if rice_data.empty:
    print("No data found for rice. Please check the Crop column for correct values.")
else:
    print("Filtered Dataset (only rice data):")
    print(rice_data.head())

    # Convert categorical columns to numerical using LabelEncoder
    label_encoder = LabelEncoder()

    # Encode each categorical column in the filtered dataframe
    for column in rice_data.select_dtypes(include=['object']).columns:
        rice_data[column] = label_encoder.fit_transform(rice_data[column].astype(str).str.strip())

    # Define features (X) and target (y) from the filtered data
    X = rice_data.drop('Yield', axis=1)  # Features (adjust column names if necessary)
    y = rice_data['Yield']               # Target (yield)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a KNN model
    knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
    knn_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = knn_model.predict(X_test)

    # Evaluate the model using Mean Squared Error (MSE) and R-squared (R²)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"KNN Mean Squared Error (MSE): {mse:.4f}")
    print(f"KNN R-squared (R²): {r2:.4f}")

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
