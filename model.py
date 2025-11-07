# %% [markdown]
# Import the neccessary libraries

# %%
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
print("----- Libraries imported successfully!")

# %% [markdown]
# Set style for better visualization

# %%
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")

# %% [markdown]
# Data Acquistion and Data Loading

# %%
# Fetch Audi stock data (Audi is part of volkswagen Group, we will use VOW3.DE)
# VOW3.DE is Volkswagen AG preference shares which include Audi
ticker = "VOW3.DE"
start_date = "2015-01-01"
end_date = "2024-01-01"

# Download the stock data
print("Downloading Audi Stock Data.....")
stock_data = yf.download(ticker,start=start_date,end=end_date)

# %%
stock_data.to_csv("data.csv",index=False)
print("----- Saved Successfully -----")

# %%
# Display basic information about the data
print(f"Data Shape: {stock_data.shape}")

# %%
stock_data

# %%
print(stock_data.info())

# %%
print("----- Basic Statistics -----")
print(stock_data.describe())

# %% [markdown]
# Data Preprocessing

# %%
# Check for missing values
stock_data_missing = stock_data.isnull().sum()
print("----- Missing Values -----")
print(stock_data_missing)

# %%
# Check for duplicated rows
stock_data_duplicated = stock_data.duplicated().sum()
print("----- Duplicated Rows -----")
print(stock_data_duplicated)

# %%
# Drop duplicates
stock_data.drop_duplicates()

# %%
print(stock_data.shape)

# %% [markdown]
# Pre-Training Visualization

# %%
# Stock Price Over Time
plt.Figure(figsize=(15,10))
plt.plot(stock_data["Close"],label="Close Price",linewidth=2)
plt.title("Audi Stock Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Volume Traded
plt.Figure(figsize=(14,7))
plt.plot(stock_data["Volume"],color="orange",alpha=0.7)
plt.title("Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Daily Returns
daily_returns = stock_data["Close"].pct_change().dropna()
plt.hist(daily_returns,bins=50,color="green",alpha=0.7,edgecolor="black")
plt.title("Distribution of Daily Returns")
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# Correlation HeatmMap
plt.figure(figsize=(14,7))
# Calculate correlation Matrix
correlation_matrix = stock_data.corr()
sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",center=0,
            square=True,linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# %%
# Moving Averages
plt.Figure(figsize=(14,7))
plt.plot(stock_data["Close"],label="Close Price",alpha=0.5)
plt.plot(stock_data["Close"].rolling(window=20).mean(),label="20-day MA",linewidth=2)
plt.plot(stock_data["Close"].rolling(window=50).mean(),label="50-day MA",linewidth=2)
plt.title("Price with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# Feature Engineering

# %%
def create_features(df):
    # Create technical indicators and features for stock prediction

    df = df.copy()

    # Technical Indicators
    df["SMA_20"] = df["Close"].rolling(window=20).mean() # Simple moving average 20 days
    df["SMA_50"] = df["Close"].rolling(window=50).mean() # Simple moving average 50 days
    df["EMA_12"] = df["Close"].ewm(span=12).mean() # Exponential moving average 12 days
    df["EMA_26"] = df["Close"].ewm(span=26).mean() # Exponential moving average 26 days

    # Volatility
    df["Volatility"] = df["Close"].rolling(window=20).std() # 20-day volatility

    # Price Rate of Change
    df["ROC"] = df["Close"].pct_change(periods=10) # 10-day rate of change

    # High-Low Percentage
    df["Pct_Change"] = df["Close"].pct_change()

    # Lag features
    df["Close_Lag_1"] = df["Close"].shift(1)
    df["Close_Lag_5"] = df["Close"].shift(5)
    df["Close_Lag_10"] = df["Close"].shift(10)

    # Volume features
    df["Volume_Change"] = df["Volume"].pct_change()

    # Drop NaN values created by rolling windows and shifts
    df = df.dropna()


    return df

# %%
# Apply feature engineering
print("Creating features........")
featured_data = create_features(stock_data)

# Define features (X) and target (y)
# We'll predict the next's day's closing price
featured_data["Target"] = featured_data["Close"].shift(-1)
featured_data = featured_data.dropna()

# Select features for modelling
feature_columns = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 
                   'EMA_12', 'EMA_26', 'Volatility', 'ROC', 
                   'Pct_Change', 'Close_Lag_1', 'Close_Lag_5', 'Close_Lag_10', 
                   'Volume_Change']

X = featured_data[feature_columns]
y = featured_data["Target"]

print(f"Final dataset shape: {X.shape}")
print(f"Feature Columns: {feature_columns}")

# %% [markdown]
# Data Splitting

# %%
# Split the data into training and testing sets (80-20 split)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,shuffle=False)

# %% [markdown]
# Data Scaling

# %%
X_train = np.clip(X_train,-1e6,1e6)
X_test = np.clip(X_test,-1e6,1e6)

# %%
print(np.isinf(X_train).sum().sum(),"Infinite values in X_train")
print(np.isnan(X_train).sum().sum(),"NaN values in X_train")

print(np.isinf(X_test).sum().sum(),"Infinite values in X_test")
print(np.isnan(X_test).sum().sum(),"NaN values in X_test")

# %%
mask_train = np.isfinite(X_train).all(axis=1)
mask_test = np.isfinite(X_test).all(axis=1)

X_train = X_train[mask_train]
X_test = X_test[mask_test]

# %%
# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# %% [markdown]
# Model Comaprison and Training

# %%
# Define all regression models to compare
models = {
    "Linear Regression":LinearRegression(),
    "Ridge Regression":Ridge(),
    "Lasso Regression":Lasso(),
    "ElasticNet":ElasticNet(),
    "Decison Tree":DecisionTreeRegressor(random_state=42),
    "Random Forest":RandomForestRegressor(random_state=42),
    #"Gradient Boosting":GradientBoostingClassifier(random_state=42),
    "SVR":SVR(),
    "K-Neighbors":KNeighborsRegressor()
}

# Dictionary to store model performance
results = {}

# %%
print("----- Training and evaluating regression models.........")
print("="*50)

# Train and evaluate each model
for name, model in models.items():
    print(f"Training {name}...")
    
    # Use scaled data for models that benefit from scaling
    if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 
                'ElasticNet', 'SVR', 'K-Neighbors']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # Train model
    model.fit(X_tr, y_train)
    
    # Make predictions
    y_pred = model.predict(X_te)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Store results
    results[name] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }
    
    print(f"{name:20} | RMSE: {rmse:8.2f} | MAE: {mae:8.2f} | R²: {r2:8.4f}")

# %% [markdown]
# Post-Training Visualization

# %%
# Create comparison DataFrame
results_df = pd.DataFrame({
    "Model":list(results.keys()),
    "RMSE":[results[model]["rmse"] for model in results],
    "MAE":[results[model]["mae"] for model in results],
    "R2_Score":[results[model]["r2"] for model in results]
})

# Sort by RMSE (lower is better)
results_df = results_df.sort_values(by="RMSE")

# %%
# Plot model comparison
fig,axes = plt.subplots(2,2,figsize=(15,12))

# Plot 1: RMSE Comparison
axes[0,0].barh(results_df["Model"],results_df["RMSE"],color="skyblue",edgecolor="black")
axes[0,0].set_xlabel("RMSE (Lower is Better)")
axes[0,0].set_title("Model Comparison - RMSE")
axes[0,0].grid(True,alpha=0.3)

# %%
# Plot 2: MAE Comparison
axes[0,1].barh(results_df["Model"],results_df["MAE"],color="lightcoral",edgecolor="black")
axes[0,1].set_xlabel("MAE (Lower is Better)")
axes[0,1].set_title("Model Comparison - MAE")
axes[0,1].grid(True,alpha=0.3)

# %%
# Plot 3: R2 Score Comparison
axes[1,0].barh(results_df["Model"],results_df["R2_Score"],color="lightgreen",edgecolor="black")
axes[1,0].set_xlabel("R2 Score (Higher is Better)")
axes[1,0].set_title("Model Comparison - R2 Score")
axes[1,0].set_xlim(0,1)
axes[1,0].grid(True,alpha=0.3)

# %%
# Plot 4: Actual vs Predicted for top 3 models
top_3_models = results_df.head(3)["Model"].values
for i,model_name in enumerate(top_3_models):
    y_pred = results[model_name]["predictions"]
    axes[1,1].scatter(y_test,y_pred,alpha=0.6,label=f"{model_name}",s=30)

axes[1,1].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"k--",lw=2)
axes[1,1].set_xlabel("Actual Prices")
axes[1,1].set_ylabel("Predicted Prices")
axes[1,1].set_title("Actual vs Predicted (Top 3 Models)")
axes[1,1].legend()
axes[1,1].grid(True,alpha=0.3)

plt.tight_layout()
plt.show()

print("Top 3 Performing Models:")
print(results_df.head())

# %% [markdown]
# Hyperparameter Tuning for the best model

# %%
# Select the best model based on RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = results[best_model_name]["model"]

print(f"Best Model: {best_model_name}")
print(f"Initial RMSE: {results[best_model_name]["rmse"]:.4f}")
print(f"Initial R2 Score: {results[best_model_name]["r2"]:.4f}")

# %%
# Define hyperparamter grids for different models
param_grids = {
    "Random Forest":{
        "n_estimators":[50,100,200],
        "max_depth":[None,10,20,30],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4]
    },
    "Gradient Boosting":{
        "n_estimators":[50,100,200],
        "learning_rate":[0.01,0.1,0.2],
        "max_depth":[3,4,5],
        "min_samples_split":[2,5]
    },
    "Decision Tree":{
        "max_depth":[None,10,20,30],
        "min_samples_split":[2,5,10],
        "min_samples_leaf":[1,2,4]
    },
    "Ridge Regression":{
        "alpha":[0.1,1.0,10.0,100.0]
    },
    "Lasso Regression":{
        "alpha":[0.1,1.0,10.0,100.0]
    },
    "ElasticNet":{
        "alpha":[0.1,1.0,10.0],
        "l1_ratio":[0.2,0.5,0.8]
    },
    "SVR":{
        "C":[0.1,1,10,100],
        "gamma":["scale","auto",0.1,0.01],
        "kernel":["rbf","linear"]
    },
    "K-Neighbors":{
        "n_neighbors":[3,5,7,9,11],
        "weights":["uniform","distance"],
        "metric":["euclidean","manhattan"]
    }
}

# %%
# Perform hyperparameter tuning if parameters are defined for the best model
if best_model_name in param_grids:
    print(f"Performing Hyperparameters Tuning for {best_model_name}..........")

    # Use appropriate data (scaled or unscaled)
    if best_model_name in ["Linear Regression","Ridge Regression","Lasso Regression","ElasticNet","SVR","K-Neighbors"]:
        X_tr,X_te = X_train_scaled,X_test_scaled

    else:
        X_tr,X_te = X_train,X_test


    # Grid search with cross-validation
    grid_search = GridSearchCV(
        best_model,
        param_grids[best_model_name],
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_tr,y_train)

    # Get best model and parameters
    tuned_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Make preddictions with tuned model
    y_pred_tuned = tuned_model.predict(X_te)

    # Calculate metrics for tuned model
    mse_tuned = mean_squared_error(y_test,y_pred_tuned)
    rmse_tuned = np.sqrt(mse_tuned)
    r2_tuned = r2_score(y_test,y_pred_tuned)

    print(f"Best Parameters: {best_params}")
    print(f"Tuned Model RMSE: {rmse_tuned:.4f}")
    print(f"Tuned Model R2 Score: {r2_tuned:.4f}")
    print(f"Improvement in RMSE: {results[best_model_name]["rmse"] - rmse_tuned:.4f}")

    # Update the best model
    best_model = tuned_model

else:
    print(f"No Hyperparameter tuning defined for {best_model_name}")
    tuned_model = best_model

# %% [markdown]
# Final Model Evaluation

# %%
# Make final predictions with the best (possibly tuned) model
if best_model_name in ["Linear Regression","Ridge Regression","Lasso Regression","ElasticNet","SVR","K-Neighbors"]:
    X_te_final = X_test_scaled
else:
    X_te_final = X_test

final_predictions = best_model.predict(X_te_final)

# %%
# Calculate final metrics
final_rmse = np.sqrt(mean_squared_error(y_test,final_predictions))
final_r2 = r2_score(y_test,final_predictions)

print("----- Final Model Performance -----")
print(f"Best Model: {best_model_name}")
print(f"Final RMSE: {final_rmse:.4f}")
print(f"Final R2 Score: {final_r2:.4f}")

# %% [markdown]
# Final Model Visualization

# %%
# Create comprehensive visualization of final results
fig,axes = plt.subplots(2,2,figsize=(15,12))

# Plot the Actual vs Predicted Prices
axes[0,0].scatter(y_test,final_predictions,alpha=0.6,color="blue")
axes[0,0].plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"r--",lw=2)
axes[0,0].set_xlabel("Actual Prices ($)")
axes[0,0].set_ylabel("Predicted Prices ($)")
axes[0,0].set_title(f"Actual vs Predicted - {best_model_name} R2 = {final_r2:.4f}")
axes[0,0].grid(True,alpha=0.3)

# Plot the time series of Actual vs Predicted
test_dates = featured_data.index[-len(y_test):]
axes[0,1].plot(test_dates,y_test.values,label="Actual",linewidth=2)
axes[0,1].plot(test_dates,final_predictions,label="Predictions",linewidth=2,alpha=0.8)
axes[0,1].set_xlabel("Date")
axes[0,1].set_ylabel("Stock Price ($)")
axes[0,1].set_title(f"Stock Price Prediction - {best_model_name}")
axes[0,1].legend()
axes[0,1].grid(True,alpha=0.3)
plt.xticks(rotation=45)

# Plot the Residuals
residuals = y_test - final_predictions
axes[1,0].scatter(final_predictions,residuals,alpha=0.6,color="green")
axes[1,0].axhline(y=0,color="red",linestyle="--")
axes[1,0].set_xlabel("Predicted Prices")
axes[1,0].set_ylabel("Residuals")
axes[1,0].set_title(f"Residual Plot - {best_model_name}")
axes[1,0].grid(True,alpha=0.3)

# Plot the Distribution of Residuals
axes[1,1].hist(residuals,bins=30,color="orange",alpha=0.7,edgecolor="black")
axes[1,1].axvline(x=0,color="red",linestyle="--",linewidth=2)
axes[1,1].set_xlabel("Residuals")
axes[1,1].set_ylabel("Frequency")
axes[1,1].set_title(f"Distribution of Residuals \n Mean {residuals.mean():.4f},Std: {residuals.std():.4f}")
axes[1,1].grid(True,alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Feature Importance 
if hasattr(best_model,"feature_importances_"):
    plt.Figure(figsize=(12,8))
    feature_importance = pd.DataFrame({
        "feature":feature_columns,
        "importance":best_model.feature_importances_
    }).sort_values("importance",ascending=False)


    plt.barh(feature_importance["feature"],feature_importance["importance"])
    plt.xlabel("Feature Importance")
    plt.title(f"Feature Importance - {best_model_name}")
    plt.grid(True,alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# Prediction Interface for new data

# %%
def predict_stock_price(model,scaler,current_data):
    """
    Predict next day's stock price using the trained model
    
    Parameters:
    model: Trained machine learning model
    scaler: Fitted StandardScaler
    current_data: Dictionary or array with current stock features
    
    Returns:
    predicted_price: Predicted stock price for the next day
    """


    # Convert input to numpy array if it's a dictionary
    if isinstance(current_data,dict):
        input_features = np.array([current_data[col] for col in feature_columns]).reshape(1,-1)
    else:
        input_features = current_data.reshape(1,-1)


    # Scale the features if needed
    model_name = type(model).__name__
    if model_name in ["LinearRegression","Ridge","Lasso","ElasticNet","SVR","KNeighborsRegressor"]:
        input_features = scaler.transform(input_features)


    # Make predictions
    prediction = model.predict(input_features)


    return prediction[0]

# %%
# Example usage with the lastest available data
print("Prediction Interface")
print("="*40)

# Get the lastest data point from the test set
latest_data = X_test.iloc[-1:].copy()


# Create a sample input for prediction
sample_input = {
    "Open":latest_data["Open"].values[0],
    "High":latest_data["High"].values[0],
    "Low":latest_data["Low"].values[0],
    "Volume":latest_data["Volume"].values[0],
    "SMA_20":latest_data["SMA_20"].values[0],
    "SMA_50":latest_data["SMA_50"].values[0],
    "EMA_12":latest_data["EMA_12"].values[0],
    "EMA_26":latest_data["EMA_26"].values[0],
    "Volatility":latest_data["Volatility"].values[0],
    "ROC":latest_data["ROC"].values[0],
    "Pct_Change":latest_data["Pct_Change"].values[0],
    "Close_Lag_1":latest_data["Close_Lag_1"].values[0],
    "Close_Lag_5":latest_data["Close_Lag_5"].values[0],
    "Close_Lag_10":latest_data["Close_Lag_10"].values[0],
    "Volume_Change":latest_data["Volume_Change"].values[0]
}

'''# Make predictions
predicted_price = predict_stock_price(best_model,scaler,sample_input)
actual_next_price = y_test.iloc[-1]

print(f"Predicted Next Day Price: ${predicted_price:.2f}")
print(f"Actual Next Day Price: ${actual_next_price:.2f}")
print(f"Prediction Error: ${abs(predicted_price - actual_next_price):.2f}")
print(f"Percentage Error: {abs((predicted_price - actual_next_price) / actual_next_price *100):.2f}")'''


'''# Convert to DataFrame with one row
sample_df = pd.DataFrame([sample_input])

# Make predictions
X_scaled = scaler.transform
predicted_price = best_model.predict(X_scaled)[0]
actual_next_price = y_test.iloc[-1]

print(f"Predicted Next Day Price: ${predicted_price:.2f}")
print(f"Actual Next Day Price: ${actual_next_price:.2f}")
print(f"Prediction Error: ${abs(predicted_price - actual_next_price):.2f}")
print(f"Percentage Error: {abs((predicted_price - actual_next_price) / actual_next_price * 100):.2f}%")'''

def predict_stock_price(model, scaler, input_data):
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_data = input_data.to_frame().T
    X_scaled = scaler.transform(input_data)
    return model.predict(X_scaled)[0]


'''# Convert to DataFrame with correct column order
sample_df = pd.DataFrame([sample_input])[X_train.columns]

# Scale using the fitted scaler (2D array expected)
X_scaled = scaler.transform(sample_df)

# Make prediction
predicted_price = best_model.predict(X_scaled)[0]
actual_next_price = y_test.iloc[-1]

# Print results
print(f"Predicted Next Day Price: ${predicted_price:.2f}")
print(f"Actual Next Day Price: ${actual_next_price:.2f}")
print(f"Prediction Error: ${abs(predicted_price - actual_next_price):.2f}")
print(f"Percentage Error: {abs((predicted_price - actual_next_price) / actual_next_price * 100):.2f}%")'''


# %% [markdown]
# Function for user input

# %%
def user_prediction_interface():
    # Interactive interface for users to input their own data and get predictions
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION INTERFACE")
    print("="*50)
    print("Enter the following stock features for prediction:")

    user_input = {}
    for feature in feature_columns:
        try:
            value = float(input(f"{feature}:"))
            user_input[feature] = value

        except ValueError:
            print(f"Invalid input for {feature}. Using default value 0.")
            user_input[feature] = 0.0


    prediction = predict_stock_price(best_model,scaler,user_input)
    print(f"Predicted Next Day Stock Price: ${prediction:.2f}")


