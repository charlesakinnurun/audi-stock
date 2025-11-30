# Audi Stock
![audi-stock](/image.jpg)

## Procedures
- Import the libaries
    - pandas
    - numpy
    - scikit-learn
    - yfinance
    - seaborn
    - matplotlib
- Data Acquisition 
    - Data Acquired from Yahoo Finance API
- Data Loading
- Data Preprocessing
    - Check for missing values
    - Check for duplicated values
    - Drop duplicates
- Pre-Training Visualization

![Stock-Price-Over-Time](/output1.png)

![Trading-Volume-Over-Time](/output2.png)

![Distribution-of-Daily-Returns](/output3.png)

![Feature-Correlation-Heatmap](/output4.png)

![Price-with-Moving-Averages](/output5.png)
- Feature Engineering
- Data Splitting
    - Split the data into training and testing sets (80-20 split)
- Data Scaling
    - Initialize the StandardScaler
- Model Comparison
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - ElasticNet Regression
    - Decison Tree Regression
    - Random Forest Regression
    - Support Vector Machine Regression (SVR)
    - K-Nearest Neighbors
- Model Training
    - Regression Model Performance Summary
    
| Model               | RMSE  | MAE  | R²      |
|---------------------|-------|-------|----------|
| Linear Regression    | 1.77 | 1.28 | 0.9128   |
| Ridge Regression     | 1.77 | 1.28 | 0.9126   |
| Lasso Regression     | 1.96 | 1.43 | 0.8932   |
| ElasticNet           | 2.15 | 1.61 | 0.8705   |
| Decision Tree        | 3.22 | 2.49 | 0.7100   |
| Random Forest        | 2.04 | 1.49 | 0.8839   |
| SVR                  | 2.37 | 1.58 | 0.8432   |
| K-Neighbors          | 3.15 | 2.22 | 0.7220   |

- Post-Training Visualization

![post-training-visualization](/output6.png)
- Hyperparameter Tuning
- Final Model Evaluation
- Final Model Visualization

![final-model-visualization](/output7.png)
- Predicted Interface for new data
- Function for User Input









## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
    - Anaconda
    - Google Colab
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```



## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/audi-stock.git
cd audi-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
bank-of-america-stock/
│
├── model.ipynb  
|── model.py    
|── BAC_stock_data.csv  
├── requirements.txt 
├── image.jpg    
├── output1.png
├── output2.png          
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── LICENSE
└── README.md          

```