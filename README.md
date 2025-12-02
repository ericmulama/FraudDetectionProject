# FraudDetectionProject

## Overview

The **FraudDetectionProject** is an end-to-end solution for detecting fraudulent financial transactions. Using machine learning, the system identifies high-risk transactions and provides a **client-friendly interactive dashboard** to monitor fraud in real-time.  

The pipeline includes:  
- Data loading and exploration  
- Feature engineering (including smart ratios to detect unusual transactions)  
- Handling class imbalance using **SMOTE**  
- Model training using **XGBoost** for high accuracy  
- Evaluation using **confusion matrix** and **classification report**  
- An interactive **Plotly Dash dashboard** to visualize predictions  

## Folder Structure

FraudDetectionProject/

├── data/

│   └── Fraud.csv

│

├── notebooks/

│   └── Fraud_Detection.ipynb        # EDA, plots, explanations

│

├── src/

│   ├── load_data.py

│   ├── eda.py

│   ├── features.py

│   ├── preprocessing/

│   │   └── preprocessing.py

│   ├── model.py

│   └── utils.py

│

├── main.py                          # Machine Learning pipeline

├── dashboard.py                     # Interactive Plotly Dash dashboard

├── requirements.txt

└── README.md


## Installation

1. Clone this repository:

git clone <repository-url>
cd FraudDetectionProject

2. Create a virtual environment and activate it:

To create vitrual environment  copy and paste "python -m venv venv" project folder.

To activate the virtual environment 

**In windows Windows**
venv\Scripts\activate

**In macOS/Linux**
source venv/bin/activate

3. Install required packages:

pip install -r requirements.txt

## Running the ML Pipeline
Make sure data/Fraud.csv exists. This is the dataset you will be working with.

Run main.py to execute the full pipeline:

"python main.py"

This will:

1. Load the dataset and check for missing values

2. Perform feature engineering (ratios, balance differences, one-hot encoding transaction types)

3. Split the dataset into training and testing sets

4. Handle class imbalance using SMOTE

5. Train an XGBoost classifier

6. Evaluate the model and print:

    - Classification report (precision, recall, F1-score)
    - Confusion matrix
    - Output example (client-friendly interpretation):
    - Out of all transactions, the model correctly identifies the majority of legitimate transactions.
    - It flags high-risk transactions (fraud) with high precision.
    - False positives are minimal, meaning clients won’t be overwhelmed with unnecessary alerts.

The model and the list of features are saved as model.pkl and features.pkl for dashboard use.

## Running the Interactive Dashboard
- Ensure model.pkl and features.pkl exist (from running main.py).
- Launch the dashboard:

Copy and paste "python dashboard.py"
- Open your browser at the URL displayed (usually http://127.0.0.1:8050/).

## Dashboard Features
1. Fraud Probability Histogram

2. Shows the distribution of predicted fraud probabilities for all transactions.

3. Helps clients see how many transactions are low-risk versus high-risk.

4. Top 10 Risky Transactions

5. Displays the accounts with the highest predicted fraud probabilities.

6. Hover over bars to see transaction details like amount and balance differences.

7. Transaction Type Pie Chart

8. Visualizes proportions of transaction types (CASH_IN, CASH_OUT, PAYMENT, TRANSFER).

9. Helps understand which types are more prone to fraud.

10. Interactive Sliders

11. Filter transactions by minimum fraud probability and minimum transaction amount.

12. Explore scenarios and focus on high-risk transactions.

## Interpreting Results
**1. Fraud Probability**

Ranges from 0 (low risk) to 1 (high risk).

Example: A transaction with a 0.9 probability is very likely fraudulent.

**2. Top Risky Transactions**

Quick way to prioritize investigations.

Clients can immediately flag or freeze suspicious accounts.

**3. Transaction Type Insights**

Understand which transaction types are most vulnerable.

Helps in designing additional safeguards or monitoring policies.

**4. Dashboard Filters**

Focus on high-value transactions or high-risk accounts.

Clients can experiment with thresholds to balance workload vs. risk.

## Dependencies
Python >= 3.8

pandas

numpy

scikit-learn

xgboost

imbalanced-learn

joblib

plotly

dash

**Install all dependencies:**

pip install -r requirements.txt

## Summary
This project provides:

- A robust machine learning model for fraud detection

- Client-ready interpretation of results

- An interactive dashboard to monitor fraud in real-time

It enables proactive detection of suspicious transactions, helping clients minimize financial loss and improve operational efficiency.
