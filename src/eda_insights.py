import pandas as pd
import json
import os

def run_eda(data_path, output_path):
    print(f"Reading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Data Cleaning
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0.0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Clean SeniorCitizen
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Convert Churn to binary numeric
    df['ChurnBinary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    insights = {}
    
    # 2. Overall Stats
    total_customers = len(df)
    churn_rate = float(df['ChurnBinary'].mean())
    total_revenue_loss = float(df[df['Churn'] == 'Yes']['MonthlyCharges'].sum())
    
    insights['overall'] = {
        'total_customers': total_customers,
        'churn_rate': churn_rate,
        'total_revenue_loss': total_revenue_loss
    }
    
    # 3. Categorical Churn Rates
    cat_columns = [
        'Contract', 'InternetService', 'PaymentMethod', 
        'TechSupport', 'OnlineSecurity', 'PaperlessBilling'
    ]
    
    insights['categorical_churn'] = {}
    for col in cat_columns:
        grouped = df.groupby(col)['ChurnBinary'].agg(['mean', 'count']) #Group by column and get churn rate and count
        insights['categorical_churn'][col] = {
            category: {
                'churn_rate': float(row['mean']),
                'count': int(row['count'])
            }
            for category, row in grouped.iterrows()
        }
        
    # 4. Tenure Binning Churn Rates ( years or 6-month blocks )
    df['TenureGroup'] = pd.cut(
        df['tenure'], 
        bins=[-1, 6, 12, 24, 36, 48, 60, 72],
        labels=['0-6 months', '6-12 months', '1-2 years', '2-3 years', '3-4 years', '4-5 years', '5-6 years']
    )
    grouped_tenure = df.groupby('TenureGroup', observed=False)['ChurnBinary'].agg(['mean', 'count'])
    insights['tenure_churn'] = {
        group: {
            'churn_rate': float(row['mean']),
            'count': int(row['count'])
        }
        for group, row in grouped_tenure.iterrows()
    }
    
    # 5. Charges vs Churn
    # Look at average monthly charges for churned vs non-churned
    charges_grouped = df.groupby('Churn')['MonthlyCharges'].mean().to_dict()
    insights['monthly_charges'] = {
        'churn_mean': float(charges_grouped.get('Yes', 0)),
        'non_churn_mean': float(charges_grouped.get('No', 0))
    }
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(insights, f, indent=4)
        
    print(f"EDA Insights successfully saved to {output_path}")

if __name__ == "__main__":
    run_eda("WA_Fn-UseC_-Telco-Customer-Churn.csv", "models/eda_insights.json")
