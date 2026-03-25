# Predictive Maintenance using Sensor Data

A complete machine learning pipeline that monitors industrial machine 
sensors and predicts equipment failure before it happens — shifting 
from reactive to predictive maintenance.

## Problem
Unplanned machine failure causes costly downtime and emergency repairs.
This project detects early warning signs from temperature and vibration 
sensors so maintenance teams can act before breakdown occurs.

## Pipeline
| Step | What it does |
|------|-------------|
| Data Simulation | Generates 500 realistic sensor readings over 1000 operating hours |
| Failure Labelling | Labels failures using 85th percentile quantile thresholds |
| Feature Engineering | Rolling averages + rate-of-change for trend and spike detection |
| EDA | Visualises degradation and compares healthy vs failed distributions |
| Rule-based Risk | Classifies risk as Low / Medium / High using engineering thresholds |
| ML Model | Logistic Regression trained on 6 features with stratified split |
| Prediction | Scores new sensor readings with a probability-based risk output |

## Tools
Python · Scikit-learn · Pandas · NumPy · Matplotlib

## Key decisions
- **Why Logistic Regression?** Outputs a probability score (not just 0/1), 
  works well on small datasets, and is easy to explain to stakeholders
- **Why stratified split?** Failures are ~10% of data — random split risks 
  putting all failures in train and none in test
- **Why quantile thresholds?** Adaptive to any data distribution, unlike 
  hardcoded values that break when data changes

## How to run
```
pip install numpy pandas scikit-learn matplotlib
jupyter notebook PredictiveMaintenance.ipynb
```

## What I'd improve with more time
- Add StandardScaler for feature normalisation
- Use time-series cross-validation instead of random split
- Try XGBoost for better recall on the minority failure class
- Build a live pipeline that computes rolling features from a real sensor stream
```

**Step 4 — Commit and push all changes**
```
git add .
git commit -m "Clean repo: remove checkpoints and CSV, improve README"
git push
```

---

## After doing this, your repo will have exactly 3 files
```
PredictiveMaintenance.ipynb   ← your clean final code
README.md                     ← professional project description  
.gitignore                    ← tells Git what to ignore forever