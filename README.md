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

| Date       | Price      | Close      | High       | Low        | Open       | Volume  |
|------------|------------|------------|------------|------------|------------|---------|
| 2015-01-02 | 103.257202 | 104.954581 | 102.182199 | 104.388788 | 104.388788 | 641,902 |
| 2015-01-05 | 99.749290  | 102.634829 | 99.296654  | 102.521672 | 102.521672 | 1,135,396 |
| 2015-01-06 | 100.371643 | 101.955865 | 98.617690  | 100.145329 | 100.145329 | 1,030,867 |
| 2015-01-07 | 100.994026 | 102.012455 | 100.060471 | 101.163766 | 101.163766 | 692,227 |
| 2015-01-08 | 104.021011 | 104.501930 | 101.276913 | 102.210467 | 102.210467 | 976,784 |
| ...        | ...        | ...        | ...        | ...        | ...        | ...     |
| 2023-12-21 | 97.592812  | 97.592812  | 96.620873  | 97.367182  | 97.367182  | 989,675 |
| 2023-12-22 | 97.679588  | 98.339118  | 97.367180  | 97.419252  | 97.419252  | 506,921 |
| 2023-12-27 | 97.887871  | 97.922577  | 96.967998  | 97.506037  | 97.506037  | 633,438 |
| 2023-12-28 | 97.332474  | 98.044070  | 96.967993  | 97.731661  | 97.731661  | 526,457 |
| 2023-12-29 | 97.020065  | 97.401893  | 96.811788  | 97.245689  | 97.245689  | 419,018 |


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
|── data.csv  
├── requirements.txt 
├── image.jpg    
├── output1.png
├── output2.png   
├── output3.png
├── output4.png
├── output5.png
├── output6.png
├── output7.png       
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── SECURITY.md
├── LICENSE
└── README.md          

```