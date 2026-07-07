# 📊 Telco Customer Churn Prediction & Retention Dashboard

An interactive Streamlit dashboard that predicts customer churn, explains why customers are likely to leave using SHAP, and helps business teams identify the most profitable retention strategies through an ROI simulator.

---

## Overview

Customer retention is one of the biggest challenges in the telecom industry. While most machine learning models simply predict whether a customer will churn, they rarely provide insights that help business teams make informed decisions.

This project goes beyond prediction by combining **machine learning**, **explainable AI**, and **business analytics** into a single dashboard. Users can identify high-risk customers, understand the reasons behind churn, estimate revenue at risk, and evaluate the financial impact of different retention campaigns.

---

## Features

- **Customer Churn Prediction**
  - Predicts whether a customer is likely to churn using trained machine learning models.
  - Supports both **XGBoost** and **Logistic Regression** models.

- **Explainable AI with SHAP**
  - Displays feature-level explanations for individual customer predictions.
  - Helps understand the key factors influencing churn.

- **ROI Campaign Simulator**
  - Simulate different retention campaigns by adjusting:
    - Offer cost
    - Campaign success rate
    - Risk threshold
  - Calculates:
    - Customers targeted
    - Revenue protected
    - Campaign cost
    - Net savings
    - ROI

- **What-If Analysis**
  - Modify customer attributes such as:
    - Contract type
    - Tech Support
    - Internet Service
    - Monthly Charges
  - Instantly observe how churn probability changes.

- **Customer Segment Explorer**
  - Filter customers based on churn risk.
  - Export high-risk customer lists as CSV for marketing or retention teams.

- **Interactive Dashboard**
  - Clean interface built with Streamlit.
  - Interactive charts powered by Plotly.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Plotly

---

## Project Structure

```text
churn-prediction-dashboard/
│
├── app.py
├── requirements.txt
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── ...
│
├── src/
│   ├── eda_insights.py
│   └── train.py
│
└── README.md
```

---

## Dataset

This project uses the **Telco Customer Churn** dataset containing customer demographics, account information, subscribed services, and churn labels.

The dataset is included in this repository.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Vismayavinodk/churn-prediction-dashboard.git
cd churn-prediction-dashboard
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

Run the following scripts once to perform exploratory analysis and train the models.

```bash
python src/eda_insights.py
python src/train.py
```

The trained models and preprocessing artifacts will be saved inside the `models/` directory.

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

## Deployment

The application can be deployed easily using **Streamlit Community Cloud**.

1. Push the repository to GitHub.
2. Ensure the `models/` folder is included.
3. Go to Streamlit Community Cloud.
4. Create a new application.
5. Select the repository and set the main file as:

```text
app.py
```

6. Deploy the application.

---

## Future Improvements

- Customer Lifetime Value (CLV) estimation
- Automated retention recommendations
- Multiple campaign strategy comparison
- Support for additional churn prediction models
- Real-time prediction using external customer data

---

## Author

**Vismaya K**

GitHub: https://github.com/Vismayavinodk
