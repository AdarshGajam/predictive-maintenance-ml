# Predictive Maintenance using Sensor Data

A machine learning pipeline that monitors temperature and vibration 
sensors to predict equipment failure before it happens.

## Tools
Python, Scikit-learn, Pandas, Matplotlib

## What it does
- Simulates realistic sensor degradation data
- Engineers rolling average and rate-of-change features
- Implements rule-based risk classification (Low/Medium/High)
- Trains a Logistic Regression model for failure prediction
- Scores new sensor readings with a probability-based risk output

## How to run
pip install numpy pandas scikit-learn matplotlib
python PredictiveMaintenance.py