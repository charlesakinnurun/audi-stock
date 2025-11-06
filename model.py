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


