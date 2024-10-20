Forest Fire Intensity Prediction
Overview
This project aims to predict forest fire intensity using various meteorological and fire weather index features from a dataset. The project employs multiple regression models to assess their effectiveness in predicting the Fire Weather Index (FWI).

Dataset
The dataset contains 243 records with features such as:
Temperature
Relative Humidity (RH)
Wind Speed (Ws)
Rainfall
FFMC, DMC, DC, ISI, BUI (Fire Weather Indices)
Classes (Fire or Not Fire)
Region
Key Statistics
Total records: 243
Fire incidents: 137 (56.4%)
Non-fire incidents: 106 (43.6%)
Features
Data Preprocessing
Dropped unnecessary columns (day, month, year).
Encoded target variable for binary classification.
Train-Test Split
Split the data into training (75%) and testing (25%) sets.
Feature Selection
Correlation analysis was conducted, retaining nine significant features.
Modeling
Implemented models: Linear Regression, Lasso Regression, Ridge Regression, and ElasticNet.
Evaluated using Mean Absolute Error (MAE) and R² Score.
Results
Best performing model: Linear Regression
Mean Absolute Error: 0.547
R² Score: 0.985
Requirements
Python 3.x
Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
