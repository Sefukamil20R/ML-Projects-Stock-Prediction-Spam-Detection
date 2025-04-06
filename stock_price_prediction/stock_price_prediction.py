import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure matplotlib to use the TkAgg backend to prevent freezing issues during plotting
matplotlib.use('TkAgg')
plt.close('all')  # Close any previously open plots to avoid conflicts
plt.ion()  # Enable interactive mode for real-time plotting

# Load the stock price dataset
file_path = "stocks/TSLA.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Overview:\n", df.head())  # Show the first 5 rows of the dataset
print("\nDataset Info:\n")
df.info()  # Display column data types and null value counts
print("\nStatistical Summary:\n", df.describe())  # Show summary statistics for numerical columns

# Convert the 'Date' column to datetime format and set it as the index for time-series analysis
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Select only the relevant columns for analysis
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# Add moving averages as new features to capture trends in stock prices
df['MA_7'] = df['Close'].rolling(window=7).mean()  # 7-day moving average
df['MA_14'] = df['Close'].rolling(window=14).mean()  # 14-day moving average
df.dropna(inplace=True)  # Remove rows with NaN values caused by rolling calculations

# Create the target variable as the next day's closing price
df['Target'] = df['Close'].shift(-1)  # Shift the 'Close' column by -1 to get the next day's price
df.dropna(inplace=True)  # Remove rows with NaN values caused by the shift operation

# Visualize the correlation between features using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')  # Annotate the heatmap with correlation values
plt.title("Feature Correlation Heatmap")
plt.show(block=False)  # Display the plot without blocking script execution
plt.pause(20)  # Pause for 3 seconds to allow the user to view the plot
plt.close()  # Close the plot window

# Scale the data to a range of 0 to 1 for better performance of machine learning models
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Separate the dataset into features (X) and target variable (y)
X = df_scaled.drop(columns=['Target'])  # Features include all columns except 'Target'
y = df_scaled['Target']  # Target variable is the next day's closing price

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)  # No shuffling to preserve time-series order

# Define the machine learning models to be used
models = {
    "Linear Regression": LinearRegression(),  # Simple linear regression
    "Ridge Regression": Ridge(),  # Linear regression with L2 regularization
    "Support Vector Regression (SVR)": SVR()  # Support vector regression
}

# Perform hyperparameter tuning for SVR using GridSearchCV
svr_params = {'C': [1, 10, 100], 'gamma': ['scale', 0.1, 0.01], 'epsilon': [0.01, 0.1, 0.5]}
svr_grid = GridSearchCV(SVR(), svr_params, cv=3, scoring='neg_mean_absolute_error')  # Use 3-fold cross-validation
svr_grid.fit(X_train, y_train)  # Fit the SVR model with different hyperparameter combinations
models["Optimized SVR"] = svr_grid.best_estimator_  # Add the best SVR model to the models dictionary

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model on the training set
    y_pred = model.predict(X_test)  # Predict on the test set
    
    # Convert predictions and actual values back to their original scale
    y_pred_actual = scaler.inverse_transform(
        np.concatenate((X_test, y_pred.reshape(-1, 1)), axis=1)
    )[:, -1]
    y_test_actual = scaler.inverse_transform(
        np.concatenate((X_test, y_test.values.reshape(-1, 1)), axis=1)
    )[:, -1]
    
    # Calculate performance metrics for the model
    mae = mean_absolute_error(y_test_actual, y_pred_actual)  # Mean Absolute Error
    mse = mean_squared_error(y_test_actual, y_pred_actual)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_test_actual, y_pred_actual)  # R² Score (coefficient of determination)
    
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}  # Store the metrics in the results dictionary
    
    # Save the trained model to a file for future use
    joblib.dump(model, f'{name.replace(" ", "_").lower()}.pkl')
    
    # Plot the actual vs. predicted stock prices
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[-len(y_test_actual):], y_test_actual, label="Actual Prices", color='blue')
    plt.plot(df.index[-len(y_pred_actual):], y_pred_actual, label=f"Predicted ({name})", linestyle='dashed', color='red')
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title(f"Stock Price Prediction - {name}")
    plt.legend()
    plt.show(block=False)  # Display the plot without blocking script execution
    plt.pause(15)  # Pause for 3 seconds to allow the user to view the plot
    plt.close()  # Close the plot window

# Print the performance metrics for each model
for model_name, metrics in results.items():
    print(f"\n📊 {model_name} Performance:")
    print(f"  - Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    print(f"  - Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    print(f"  - Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    print(f"  - R² Score: {metrics['R2 Score']:.2f}")