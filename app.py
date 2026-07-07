import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.express as px
import plotly.graph_objects as go
import shap

# 1. Page Configuration & Custom CSS Styling
st.set_page_config(
    page_title="Telco Churn Intelligence & Retention Deck",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Injection for modern visual design
def inject_custom_css():
    st.markdown("""
        <style>
        /* Base styles */
        .main {
            background-color: #0A0E17;
            color: #F1F5F9;
        }
        
        /* Metric Card styling */
        div[data-testid="metric-container"] {
            background-color: #131C2E;
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            border-color: #00D2FF;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00D2FF;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }
        
        /* App Title Header Banner */
        .title-banner {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border-left: 5px solid #00D2FF;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.25);
        }
        .title-banner h1 {
            color: #FFFFFF;
            font-size: 2.2rem;
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.025em;
        }
        .title-banner p {
            color: #94A3B8;
            font-size: 1.05rem;
            margin: 5px 0 0 0;
        }
        
        /* Custom card class */
        .custom-card {
            background-color: #131C2E;
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .custom-card h3 {
            margin-top: 0;
            color: #00D2FF;
            font-size: 1.2rem;
        }
        
        /* Highlight Info box styling */
        div.stAlert {
            background-color: #1E293B;
            border: 1px solid #334155;
            color: #F1F5F9;
            border-radius: 8px;
        }
        
        /* Tabs styling */
        button[data-baseweb="tab"] {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            color: #94A3B8 !important;
            padding: 12px 20px !important;
            background-color: transparent !important;
            border: none !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00D2FF !important;
            border-bottom: 2px solid #00D2FF !important;
        }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# 2. Cache Data & Model Loading
@st.cache_resource
def load_ml_assets():
    # Load scikit-learn preprocessor pipeline
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    # Load Logistic Regression model
    with open("models/lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
    # Load XGBoost model
    with open("models/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    # Load SHAP TreeExplainer
    with open("models/shap_explainer.pkl", "rb") as f:
        shap_explainer = pickle.load(f)
    # Load precalculated SHAP values
    with open("models/shap_test_data.pkl", "rb") as f:
        shap_test_data = pickle.load(f)
        
    return preprocessor, lr_model, xgb_model, shap_explainer, shap_test_data

@st.cache_data
def load_raw_data_and_predict():
    # Load raw dataset
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = df['TotalCharges'].replace(' ', '0.0')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    
    # Load models dynamically for scoring the full dataset
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("models/xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("models/lr_model.pkl", "rb") as f:
        lr_model = pickle.load(f)
        
    # Standard feature selection (matches training structure)
    X = df.drop(columns=['customerID', 'Churn'])
    X_proc = preprocessor.transform(X)
    
    # Predict probabilities
    df['XGB_Probability'] = xgb_model.predict_proba(X_proc)[:, 1]
    df['LR_Probability'] = lr_model.predict_proba(X_proc)[:, 1]
    
    # Risk classification category
    df['RiskCategory'] = pd.cut(
        df['XGB_Probability'],
        bins=[-0.1, 0.3, 0.7, 1.01],
        labels=['Low Risk (<30%)', 'Medium Risk (30-70%)', 'High Risk (>70%)']
    )
    return df

@st.cache_data
def load_eda_insights():
    if os.path.exists("models/eda_insights.json"):
        with open("models/eda_insights.json", "r") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_metrics():
    if os.path.exists("models/metrics.json"):
        with open("models/metrics.json", "r") as f:
            return json.load(f)
    return {}

# Load all assets
try:
    preprocessor, lr_model, xgb_model, shap_explainer, shap_test_data = load_ml_assets()
    df = load_raw_data_and_predict()
    eda_insights = load_eda_insights()
    model_metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading models or data. Please verify that the training script has run successfully. Details: {e}")
    st.stop()

# Helper: Clean feature names for SHAP and display
def get_clean_feature_info(feature_name, preprocessed_value, raw_row):
    if "_" in feature_name:
        parts = feature_name.split("_")
        base_feature = parts[0]
        category = "_".join(parts[1:])
        
        raw_val = raw_row[base_feature].values[0] if base_feature in raw_row.columns else None
        
        # If this is the active category for this customer
        if str(raw_val).strip() == category.strip():
            display_name = f"{base_feature}: {category}"
        else:
            display_name = f"{base_feature}: Not {category}"
    else:
        raw_val = raw_row[feature_name].values[0] if feature_name in raw_row.columns else preprocessed_value
        if feature_name == 'tenure':
            display_name = f"Tenure: {int(raw_val)} months"
        elif feature_name == 'MonthlyCharges':
            display_name = f"Monthly Charges: ${raw_val:.2f}"
        elif feature_name == 'TotalCharges':
            display_name = f"Total Charges: ${raw_val:.2f}"
        else:
            display_name = f"{feature_name}: {raw_val}"
            
    return display_name

# Header Banner
st.markdown("""
    <div class="title-banner">
        <h1>📊 Telco Churn Intelligence & Retention Console</h1>
        <p>Explain churn risk with SHAP explainability and optimize retention strategy ROI in real-time.</p>
    </div>
""", unsafe_allow_html=True)

# 3. Sidebar Panel Configuration
st.sidebar.markdown("### ⚙️ Model & Target Selector")
selected_model_name = st.sidebar.selectbox(
    "Prediction Engine",
    ["XGBoost (Challenger)", "Logistic Regression (Baseline)"],
    index=0
)
model_key = "XGB_Probability" if "XGBoost" in selected_model_name else "LR_Probability"
active_model = xgb_model if "XGBoost" in selected_model_name else lr_model

# Retention Campaign Parameters
st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Retention Campaign Settings")
offer_cost = st.sidebar.slider(
    "Retention Offer Cost ($/month)",
    min_value=5,
    max_value=50,
    value=15,
    step=5,
    help="Cost of the discount/retention incentive given to high-risk customers per month."
)
success_rate = st.sidebar.slider(
    "Offer Acceptance Rate (%)",
    min_value=5,
    max_value=100,
    value=40,
    step=5,
    help="Percentage of targeted at-risk customers who accept the retention incentive and stay."
) / 100.0

risk_threshold = st.sidebar.slider(
    "Targeting Risk Threshold (%)",
    min_value=10,
    max_value=90,
    value=50,
    step=5,
    help="Target customers whose predicted probability of churn is greater than or equal to this threshold."
) / 100.0

# 4. App Tabs Layout
tab1, tab2, tab3 = st.tabs([
    "📊 Executive Summary & ROI Simulator", 
    "🔍 Segment Explorer", 
    "👤 Individual Drill-Down & Sandbox"
])

# ----------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY & ROI SIMULATOR
# ----------------------------------------------------
with tab1:
    st.markdown("### 📊 Business Impact and ROI Simulation")
    st.markdown(
        "This section maps the machine learning churn probabilities to real financial metrics, "
        "enabling stakeholders to understand the economic impact of targeting strategies."
    )
    
    # Calculations based on active model
    df['Prob'] = df[model_key]
    targeted_mask = df['Prob'] >= risk_threshold
    num_targeted = targeted_mask.sum()
    pct_targeted = num_targeted / len(df)
    
    # Financial calculations
    actual_churn_mask = df['Churn'] == 'Yes'
    targeted_churners = targeted_mask & actual_churn_mask
    targeted_non_churners = targeted_mask & (~actual_churn_mask)
    
    # 1. Total baseline revenue loss per month from actual churners
    baseline_loss = df[actual_churn_mask]['MonthlyCharges'].sum()
    
    # 2. Saved churn revenue
    # The success rate represents how many TP accept and stay. We save their monthly charges minus the offer cost.
    tp_charges = df[targeted_churners]['MonthlyCharges'].sum()
    recovered_revenue = success_rate * tp_charges
    
    # 3. Campaign Cost
    # We pay the offer cost for targeted non-churners (FP) and targeted churners who accept the offer (TP * SuccessRate)
    num_tp_accept = targeted_churners.sum() * success_rate
    num_fp_accept = targeted_non_churners.sum() # FP accept the discount since they are staying anyway
    campaign_cost = offer_cost * (num_fp_accept + num_tp_accept)
    
    # 4. Net Monthly Savings
    net_savings = recovered_revenue - campaign_cost
    
    # 5. Campaign ROI
    campaign_roi = (net_savings / campaign_cost * 100) if campaign_cost > 0 else 0.0
    
    # Render KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Baseline Revenue Loss",
            value=f"${baseline_loss:,.2f}/mo",
            delta=None,
            help="Total monthly charges of actual churned customers if no retention campaign is executed."
        )
    with col2:
        st.metric(
            label="Recovered Revenue",
            value=f"${recovered_revenue:,.2f}/mo",
            delta=None,
            help="Projected revenue saved from churners who accept the offer and stay."
        )
    with col3:
        st.metric(
            label="Monthly Campaign Cost",
            value=f"${campaign_cost:,.2f}/mo",
            delta=None,
            help="Total cost of offering discounts (given to false alarms + rescued customers)."
        )
    with col4:
        st.metric(
            label="Net Campaign Savings",
            value=f"${net_savings:,.2f}/mo",
            delta=f"{campaign_roi:+.1f}% ROI",
            delta_color="normal" if net_savings >= 0 else "inverse",
            help="Net financial impact: Recovered Revenue minus Campaign Cost."
        )
        
    st.markdown("---")
    
    # Layout split: Optimization Curve and Statistics
    col_chart, col_text = st.columns([2, 1])
    
    with col_chart:
        # Dynamic calculation of ROI Optimization Curve across all thresholds
        pct_list = []
        net_savings_list = []
        costs_list = []
        threshold_list = []
        
        # Calculate for top percentiles (0% to 100%)
        df_sorted = df.sort_values('Prob', ascending=False).reset_index(drop=True)
        n_total = len(df_sorted)
        
        for pct in range(0, 101, 2):
            n_target = int(n_total * (pct / 100))
            if n_target == 0:
                pct_list.append(0.0)
                net_savings_list.append(0.0)
                costs_list.append(0.0)
                threshold_list.append(1.0)
                continue
                
            targeted_sub = df_sorted.iloc[:n_target]
            is_c = targeted_sub['Churn'] == 'Yes'
            is_nc = targeted_sub['Churn'] == 'No'
            
            tp_sub = is_c.sum()
            fp_sub = is_nc.sum()
            
            charges_tp_sub = targeted_sub[is_c]['MonthlyCharges'].sum()
            
            recovered_sub = success_rate * charges_tp_sub
            cost_sub = offer_cost * (fp_sub + success_rate * tp_sub)
            savings_sub = recovered_sub - cost_sub
            
            thresh = float(df_sorted.iloc[n_target-1]['Prob'])
            
            pct_list.append(pct)
            net_savings_list.append(savings_sub)
            costs_list.append(cost_sub)
            threshold_list.append(thresh)
            
        roi_df = pd.DataFrame({
            'PercentTargeted': pct_list,
            'RiskThreshold': threshold_list,
            'NetSavings': net_savings_list,
            'CampaignCost': costs_list
        })
        
        # Find optimal targeting percentile
        opt_idx = roi_df['NetSavings'].idxmax()
        opt_pct = roi_df.loc[opt_idx, 'PercentTargeted']
        opt_thresh = roi_df.loc[opt_idx, 'RiskThreshold']
        opt_savings = roi_df.loc[opt_idx, 'NetSavings']
        
        # Plot Plotly chart
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=roi_df['PercentTargeted'], y=roi_df['NetSavings'],
            mode='lines', name='Net Monthly Savings ($)',
            line=dict(color='#00D2FF', width=3)
        ))
        fig_roi.add_trace(go.Scatter(
            x=roi_df['PercentTargeted'], y=roi_df['CampaignCost'],
            mode='lines', name='Campaign Cost ($)',
            line=dict(color='#94A3B8', width=1.5, dash='dash')
        ))
        
        # Vertical Line at Optimal targeting
        fig_roi.add_vline(
            x=opt_pct, line_width=2, line_dash="dash", line_color="#FF4B4B",
            annotation_text=f"Optimal Target: Top {opt_pct}%<br>Threshold: {opt_thresh:.0%}", 
            annotation_position="top left"
        )
        
        fig_roi.update_layout(
            title="Campaign ROI Optimization Curve",
            xaxis_title="% of Highest-Risk Customers Targeted",
            yaxis_title="Monthly Financial Impact ($)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F1F5F9"),
            xaxis=dict(showgrid=True, gridcolor="#1E293B"),
            yaxis=dict(showgrid=True, gridcolor="#1E293B")
        )
        st.plotly_chart(fig_roi, use_container_width=True)
        
    with col_text:
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("### 💡 Business Recommendation")
        
        # Calculate optimal stats
        opt_cost = roi_df.loc[opt_idx, 'CampaignCost']
        opt_roi = (opt_savings / opt_cost * 100) if opt_cost > 0 else 0.0
        
        st.markdown(f"""
            Based on the current scenario simulator parameters:
            - **Optimal Targeting Strategy**: Target the top **{opt_pct}%** of highest-risk customers.
            - **Operating Risk Threshold**: Churn probability of **$\ge$ {opt_thresh:.1%}**.
            - **Max Net Savings**: Projected **${opt_savings:,.2f}** saved per month in net revenue.
            - **Campaign Budget**: Required spend of **${opt_cost:,.2f}/month** on retention offers.
            - **Target Return on Investment**: Expected campaign ROI of **{opt_roi:.1f}%**.
        """)
        
        st.info(
            f"Currently, you are targeting **{pct_targeted:.1%}** of the customer base (Threshold: {risk_threshold:.1%}). "
            f"Adjusting your threshold to {opt_thresh:.1%} would increase net monthly savings "
            f"by **${(opt_savings - net_savings):+,.2f}**."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Historical Model performance info
        st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
        st.markdown("### 📈 Model Evaluation Metrics")
        
        # Display model metrics from train.py
        if model_metrics:
            metrics_display = model_metrics.get("XGBoost" if "XGBoost" in selected_model_name else "Logistic Regression", {})
            st.markdown(f"**Current Model: {selected_model_name}**")
            st.markdown(f"- **Recall (Sensitivity)**: `{metrics_display.get('recall', 0.0):.2%}` (captures {metrics_display.get('recall', 0.0):.1%} of churners)")
            st.markdown(f"- **Precision (Targeting Efficiency)**: `{metrics_display.get('precision', 0.0):.2%}`")
            st.markdown(f"- **F1-Score**: `{metrics_display.get('f1', 0.0):.2%}`")
            st.markdown(f"- **ROC-AUC**: `{metrics_display.get('roc_auc', 0.0):.4f}`")
        else:
            st.markdown("Model metrics not loaded.")
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 2: SEGMENT EXPLORER
# ----------------------------------------------------
with tab2:
    st.markdown("### 🔍 Customer Segment Explorer")
    st.markdown(
        "Slice and dice the customer base to locate specific demographics or product tiers "
        "that present anomalous churn risk. Apply filters below to isolate customer subgroups."
    )
    
    # Row filters layout
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        selected_contracts = st.multiselect(
            "Contract Type",
            options=df['Contract'].unique().tolist(),
            default=df['Contract'].unique().tolist()
        )
    with col_f2:
        selected_internet = st.multiselect(
            "Internet Service Type",
            options=df['InternetService'].unique().tolist(),
            default=df['InternetService'].unique().tolist()
        )
    with col_f3:
        selected_support = st.multiselect(
            "Tech Support Enrollment",
            options=df['TechSupport'].unique().tolist(),
            default=df['TechSupport'].unique().tolist()
        )
    with col_f4:
        selected_payment = st.multiselect(
            "Payment Method",
            options=df['PaymentMethod'].unique().tolist(),
            default=df['PaymentMethod'].unique().tolist()
        )
        
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        tenure_range = st.slider(
            "Tenure Range (Months)",
            min_value=int(df['tenure'].min()),
            max_value=int(df['tenure'].max()),
            value=(int(df['tenure'].min()), int(df['tenure'].max()))
        )
    with col_s2:
        charges_range = st.slider(
            "Monthly Charges Range ($)",
            min_value=float(df['MonthlyCharges'].min()),
            max_value=float(df['MonthlyCharges'].max()),
            value=(float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max()))
        )
        
    # Apply segment masks
    segment_mask = (
        df['Contract'].isin(selected_contracts) &
        df['InternetService'].isin(selected_internet) &
        df['TechSupport'].isin(selected_support) &
        df['PaymentMethod'].isin(selected_payment) &
        df['tenure'].between(tenure_range[0], tenure_range[1]) &
        df['MonthlyCharges'].between(charges_range[0], charges_range[1])
    )
    
    segment_df = df[segment_mask]
    
    if segment_df.empty:
        st.warning("⚠️ No customers match the selected segment filters. Please expand your filtering criteria.")
    else:
        # Segment summary KPIs
        total_baseline_churn_rate = (df['Churn'] == 'Yes').mean()
        segment_churn_rate = (segment_df['Churn'] == 'Yes').mean()
        segment_avg_prob = segment_df['Prob'].mean()
        segment_total_revenue = segment_df['MonthlyCharges'].sum()
        segment_revenue_at_risk = (segment_df['Prob'] * segment_df['MonthlyCharges']).sum()
        
        col_seg1, col_seg2, col_seg3, col_seg4 = st.columns(4)
        with col_seg1:
            st.metric(
                label="Segment Size",
                value=f"{len(segment_df):,} customers",
                delta=f"{len(segment_df)/len(df):.1%} of base"
            )
        with col_seg2:
            st.metric(
                label="Historical Churn Rate",
                value=f"{segment_churn_rate:.1%}",
                delta=f"{(segment_churn_rate - total_baseline_churn_rate):+.1%} vs Base Churn ({total_baseline_churn_rate:.1%})",
                delta_color="inverse"
            )
        with col_seg3:
            st.metric(
                label="Average Model Risk",
                value=f"{segment_avg_prob:.1%}",
                delta=None,
                help="Mean predicted probability of churn across this segment."
            )
        with col_seg4:
            st.metric(
                label="Revenue at Churn Risk",
                value=f"${segment_revenue_at_risk:,.2f}/mo",
                delta=None,
                help="Sum of monthly charges weighted by predicted churn probabilities."
            )
            
        st.markdown("---")
        
        # Segment charts
        col_seg_chart1, col_seg_chart2 = st.columns(2)
        
        with col_seg_chart1:
            # Risk Category Distribution inside segment
            risk_counts = segment_df['RiskCategory'].value_counts().reset_index()
            risk_counts.columns = ['RiskTier', 'Count']
            
            # Ensure all tiers are plotted
            order_mapping = {'Low Risk (<30%)': 0, 'Medium Risk (30-70%)': 1, 'High Risk (>70%)': 2}
            risk_counts['Sort'] = risk_counts['RiskTier'].map(order_mapping)
            risk_counts = risk_counts.sort_values('Sort')
            
            fig_dist = px.bar(
                risk_counts, x='RiskTier', y='Count',
                title="Segment Churn Risk Profile Distribution",
                color='RiskTier',
                color_discrete_map={
                    'Low Risk (<30%)': '#00D2FF',
                    'Medium Risk (30-70%)': '#F59E0B',
                    'High Risk (>70%)': '#EF4444'
                }
            )
            fig_dist.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F1F5F9"),
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#1E293B")
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with col_seg_chart2:
            # Average Tenure vs Monthly Charges Scatter colored by risk probability
            fig_scat = px.scatter(
                segment_df, x='tenure', y='MonthlyCharges',
                color='Prob',
                color_continuous_scale=px.colors.sequential.Bluered,
                labels={'tenure': 'Tenure (Months)', 'MonthlyCharges': 'Monthly Charges ($)', 'Prob': 'Churn Risk'},
                title="Customer Distribution: Tenure vs. Monthly Charges"
            )
            fig_scat.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F1F5F9"),
                xaxis=dict(showgrid=True, gridcolor="#1E293B"),
                yaxis=dict(showgrid=True, gridcolor="#1E293B")
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            
        st.markdown("### 📋 High-Risk Customer Target Roster")
        st.markdown(
            "Below are the top 25 highest-risk customers in your selected segment. "
            "You can download this roster as a CSV to feed into downstream email or phone campaign channels."
        )
        
        # Sort by active probability
        roster_df = segment_df.sort_values('Prob', ascending=False).head(25)[
            ['customerID', 'gender', 'tenure', 'Contract', 'InternetService', 'TechSupport', 'MonthlyCharges', 'Prob', 'RiskCategory']
        ]
        roster_df['Prob'] = roster_df['Prob'].apply(lambda x: f"{x:.1%}")
        roster_df = roster_df.rename(columns={
            'customerID': 'Customer ID',
            'gender': 'Gender',
            'tenure': 'Tenure (Mo)',
            'Contract': 'Contract',
            'InternetService': 'Internet',
            'TechSupport': 'Tech Support',
            'MonthlyCharges': 'Monthly Charges',
            'Prob': 'Churn Prob',
            'RiskCategory': 'Risk Tier'
        })
        
        st.dataframe(roster_df, use_container_width=True, hide_index=True)
        
        # CSV download button
        csv_data = segment_df.sort_values('Prob', ascending=False).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Segment Roster (CSV)",
            data=csv_data,
            file_name="telco_high_risk_segment_roster.csv",
            mime="text/csv"
        )

# ----------------------------------------------------
# TAB 3: INDIVIDUAL DRILL-DOWN & SANDBOX
# ----------------------------------------------------
with tab3:
    st.markdown("### 👤 Individual Customer Churn Analysis")
    st.markdown(
        "Select a customer to view their demographic, product profiles, risk score, and "
        "explainable AI drivers. Modify their features in the What-If Sandbox to see how offering "
        "discounts or contract modifications affects their churn risk."
    )
    
    # Customer Search panel divided by risk category for easy inspection
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        high_risk_ids = df[df['XGB_Probability'] >= 0.75]['customerID'].head(10).tolist()
        cust_high = st.selectbox("🔥 Sample High Risk Customers (risk > 75%)", options=high_risk_ids)
    with col_c2:
        med_risk_ids = df[df['XGB_Probability'].between(0.4, 0.6)]['customerID'].head(10).tolist()
        cust_med = st.selectbox("🟡 Sample Medium Risk Customers (risk 40-60%)", options=med_risk_ids)
    with col_c3:
        low_risk_ids = df[df['XGB_Probability'] <= 0.12]['customerID'].head(10).tolist()
        cust_low = st.selectbox("🟢 Sample Low Risk Customers (risk < 12%)", options=low_risk_ids)
        
    search_col_1, search_col_2 = st.columns([2, 1])
    with search_col_1:
        typed_cust_id = st.text_input("🔍 Or Enter Custom Customer ID manually (e.g. 7590-VHVEG):").strip()
    
    # Determine chosen ID
    selected_cust_id = typed_cust_id if typed_cust_id else None
    if not selected_cust_id:
        # Use one of the samples. Default to the high-risk selected one if user hasn't typed anything
        # We can use a radio button or buttons to switch, or just take whichever dropdown was changed last.
        # To keep it simple and robust, we check if typed_cust_id is active. If not, we provide buttons or a selector.
        # Let's add a choice option:
        selected_cust_id = st.radio(
            "Use Customer ID from:",
            options=["High Risk Sample", "Medium Risk Sample", "Low Risk Sample"],
            horizontal=True
        )
        if selected_cust_id == "High Risk Sample":
            selected_cust_id = cust_high
        elif selected_cust_id == "Medium Risk Sample":
            selected_cust_id = cust_med
        else:
            selected_cust_id = cust_low
            
    # Load customer details
    row = df[df['customerID'] == selected_cust_id]
    
    if row.empty:
        st.error(f"❌ Customer ID '{selected_cust_id}' not found in the dataset. Please enter a valid ID.")
    else:
        st.markdown(f"## 👤 Analysis for Customer: `{selected_cust_id}`")
        
        # Grid of Info
        col_grid1, col_grid2 = st.columns([1, 1])
        
        with col_grid1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("### 📋 Demographic & Billing Details")
            st.markdown(f"""
                - **Gender**: `{row['gender'].values[0]}`
                - **Senior Citizen**: `{row['SeniorCitizen'].values[0]}`
                - **Partner**: `{row['Partner'].values[0]}` | **Dependents**: `{row['Dependents'].values[0]}`
                - **Tenure**: `{row['tenure'].values[0]} months`
                - **Contract**: `{row['Contract'].values[0]}`
                - **Paperless Billing**: `{row['PaperlessBilling'].values[0]}`
                - **Payment Method**: `{row['PaymentMethod'].values[0]}`
                - **Monthly Charges**: `${row['MonthlyCharges'].values[0]:.2f}`
                - **Total Charges**: `${row['TotalCharges'].values[0]:.2f}`
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("### 🛠️ Product & Service Subscriptions")
            st.markdown(f"""
                - **Phone Service**: `{row['PhoneService'].values[0]}`
                - **Multiple Lines**: `{row['MultipleLines'].values[0]}`
                - **Internet Service**: `{row['InternetService'].values[0]}`
                - **Online Security**: `{row['OnlineSecurity'].values[0]}`
                - **Online Backup**: `{row['OnlineBackup'].values[0]}`
                - **Device Protection**: `{row['DeviceProtection'].values[0]}`
                - **Tech Support**: `{row['TechSupport'].values[0]}`
                - **Streaming TV**: `{row['StreamingTV'].values[0]}`
                - **Streaming Movies**: `{row['StreamingMovies'].values[0]}`
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_grid2:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("### 🚨 Churn Risk Analysis")
            
            # Predict
            prob_xgb = row['XGB_Probability'].values[0]
            prob_lr = row['LR_Probability'].values[0]
            prob_val = prob_xgb if "XGBoost" in selected_model_name else prob_lr
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_val * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Churn Probability (%)", 'font': {'color': '#F1F5F9', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#F1F5F9"},
                    'bar': {'color': "#EF4444" if prob_val > 0.7 else "#F59E0B" if prob_val > 0.3 else "#00D2FF"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "#1E293B",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.15)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.15)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                    ]
                }
            ))
            fig_gauge.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F1F5F9")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Risk Category Alert Box
            risk_tier = row['RiskCategory'].values[0]
            if "High" in risk_tier:
                st.error(f"🔴 **HIGH CHURN RISK DETECTED** ({prob_val:.1%}). Immediate proactive retention targeting is recommended.")
            elif "Medium" in risk_tier:
                st.warning(f"🟡 **MEDIUM CHURN RISK DETECTED** ({prob_val:.1%}). Customer displays signs of churn vulnerability. Offer upgrades.")
            else:
                st.success(f"🟢 **LOW CHURN RISK DETECTED** ({prob_val:.1%}). Customer displays strong retention loyalty profile.")
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Explainability & Sandbox row
        col_exp, col_sand = st.columns([1.1, 0.9])
        
        with col_exp:
            st.markdown("### 🧬 AI Explainability (SHAP Risk Drivers)")
            st.markdown(
                "This chart explains *why* the machine learning model outputted this specific risk score. "
                "**Cyan bars** pull the risk score downwards (retaining factors), while **Red bars** pull the risk score upwards (churn risk factors)."
            )
            
            # Prepare row for SHAP
            X_cust = row.drop(columns=['customerID', 'Churn', 'XGB_Probability', 'LR_Probability', 'RiskCategory', 'Prob'], errors='ignore')
            X_cust_proc = preprocessor.transform(X_cust)
            
            # Calculate SHAP values dynamically using loaded TreeExplainer (XGBoost is explainable by TreeExplainer)
            try:
                shap_explanation = shap_explainer(X_cust_proc)
                shap_vals = shap_explanation.values[0]
                
                # Zip and pair features
                feature_names = preprocessor.feature_names_
                shap_data = []
                for name, val, proc_val in zip(feature_names, shap_vals, X_cust_proc[0]):
                    if abs(val) > 0.005:  # Keep meaningful SHAP contributions
                        clean_label = get_clean_feature_info(name, proc_val, X_cust)
                        shap_data.append({
                            'feature': name,
                            'clean_label': clean_label,
                            'shap_val': val,
                            'abs_shap': abs(val)
                        })
                
                shap_df = pd.DataFrame(shap_data)
                
                if shap_df.empty:
                    st.info("No significant features identified for this customer.")
                else:
                    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(10) # Get top 10
                    shap_df = shap_df.sort_values('shap_val', ascending=True) # Sort signed value for nice visual stack
                    
                    # Colors: Red for positive (risk increase), Cyan for negative (risk decrease)
                    shap_df['color'] = np.where(shap_df['shap_val'] > 0, '#EF4444', '#00D2FF')
                    
                    fig_shap = go.Figure()
                    fig_shap.add_trace(go.Bar(
                        y=shap_df['clean_label'],
                        x=shap_df['shap_val'],
                        orientation='h',
                        marker_color=shap_df['color'],
                        hovertemplate="Feature: %{y}<br>Impact (log-odds): %{x:+.3f}<extra></extra>"
                    ))
                    fig_shap.update_layout(
                        title=f"SHAP Explanations for Customer {selected_cust_id}",
                        xaxis_title="Impact on Churn Likelihood (Log-odds scale)",
                        yaxis_title="",
                        height=380,
                        margin=dict(l=10, r=20, t=45, b=20),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#F1F5F9"),
                        xaxis=dict(
                            showgrid=True,
                            gridcolor="#1E293B",
                            zeroline=True,
                            zerolinecolor="#64748B"
                        ),
                        yaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.warning(f"Unable to calculate individual SHAP chart. Details: {e}")
                
        with col_sand:
            st.markdown("### 🎛️ What-If Retention Sandbox")
            st.markdown(
                "Simulate offering various contract modifications, discount offers, or security service packages "
                "to find the most cost-effective retention strategy for this specific customer."
            )
            
            # Interactive sandbox options
            current_contract = row['Contract'].values[0]
            contract_options = ['Month-to-month', 'One year', 'Two year']
            sim_contract = st.selectbox(
                "📝 Contract Offer",
                options=contract_options,
                index=contract_options.index(current_contract)
            )
            
            current_support = row['TechSupport'].values[0]
            support_options = ['No', 'Yes', 'No internet service']
            sim_support = st.selectbox(
                "🛠️ Add Tech Support",
                options=support_options,
                index=support_options.index(current_support)
            )
            
            current_security = row['OnlineSecurity'].values[0]
            security_options = ['No', 'Yes', 'No internet service']
            sim_security = st.selectbox(
                "🔒 Add Online Security",
                options=security_options,
                index=security_options.index(current_security)
            )
            
            current_backup = row['OnlineBackup'].values[0]
            backup_options = ['No', 'Yes', 'No internet service']
            sim_backup = st.selectbox(
                "☁️ Add Online Backup",
                options=backup_options,
                index=backup_options.index(current_backup)
            )
            
            current_payment = row['PaymentMethod'].values[0]
            payment_options = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
            sim_payment = st.selectbox(
                "💳 Payment Method",
                options=payment_options,
                index=payment_options.index(current_payment)
            )
            
            current_charges = float(row['MonthlyCharges'].values[0])
            # Allow discounts by reducing charges
            sim_charges = st.slider(
                "💸 Adjust Monthly Charges ($)",
                min_value=max(10.0, current_charges - 40.0),
                max_value=current_charges + 10.0,
                value=current_charges,
                step=1.0,
                help="Decrease monthly charges to simulate a loyalty discount."
            )
            
            # Construct modified customer row
            sim_row = X_cust.copy()
            sim_row['Contract'] = sim_contract
            sim_row['TechSupport'] = sim_support
            sim_row['OnlineSecurity'] = sim_security
            sim_row['OnlineBackup'] = sim_backup
            sim_row['PaymentMethod'] = sim_payment
            sim_row['MonthlyCharges'] = sim_charges
            
            # Adjust TotalCharges proportionally
            sim_row['TotalCharges'] = max(0.0, float(row['TotalCharges'].values[0]) - current_charges + sim_charges)
            
            # Reprocess and run prediction
            X_sim_proc = preprocessor.transform(sim_row)
            sim_prob = active_model.predict_proba(X_sim_proc)[0, 1]
            
            # Display simulated results
            st.markdown("<div style='background-color:#131C2E; padding:15px; border-radius:8px; border:1px solid #1E293B;'>", unsafe_allow_html=True)
            st.markdown(f"#### 📊 Simulation Results:")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown(f"**Baseline Risk**: `{prob_val:.1%}`")
                st.markdown(f"**Simulated Risk**: `{sim_prob:.1%}`")
            with col_res2:
                prob_diff = sim_prob - prob_val
                if prob_diff <= -0.05:
                    st.markdown(f"<span style='color:#10B981; font-weight:bold; font-size:1.15rem;'>📉 Risk Reduced by {abs(prob_diff):.1%}!</span>", unsafe_allow_html=True)
                elif prob_diff >= 0.05:
                    st.markdown(f"<span style='color:#EF4444; font-weight:bold; font-size:1.15rem;'>📈 Risk Increased by {prob_diff:.1%}!</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:#94A3B8; font-weight:bold;'>No major risk change</span>", unsafe_allow_html=True)
            
            # Auto-generated prescriptions
            if prob_val > 0.4 and sim_prob < 0.25:
                discount_amount = current_charges - sim_charges
                discount_str = f" offering a loyalty discount of **${discount_amount:.2f}/mo**" if discount_amount > 0 else ""
                contract_str = f" transitioning the customer to a **{sim_contract}** contract" if sim_contract != current_contract else ""
                services_added = []
                if sim_support == 'Yes' and current_support != 'Yes': services_added.append("Tech Support")
                if sim_security == 'Yes' and current_security != 'Yes': services_added.append("Online Security")
                if sim_backup == 'Yes' and current_backup != 'Yes': services_added.append("Online Backup")
                services_str = f" enrolling them in **{', '.join(services_added)}**" if services_added else ""
                
                conjunctions = [x for x in [discount_str, contract_str, services_str] if x != ""]
                presc_text = "Actionable Retention Recipe: Successfully mitigated churn risk by"
                if len(conjunctions) == 1:
                    presc_text += conjunctions[0] + "."
                elif len(conjunctions) == 2:
                    presc_text += conjunctions[0] + " and" + conjunctions[1] + "."
                elif len(conjunctions) >= 3:
                    presc_text += conjunctions[0] + "," + conjunctions[1] + ", and" + conjunctions[2] + "."
                else:
                    presc_text += " adjusting account features."
                    
                st.success(f"💡 **Recommended Plan**:<br>{presc_text}", icon="✅")
            elif prob_val > 0.4:
                st.info("💡 **Prescription Hint**: Month-to-month contract structure, Fiber Optic internet, and lack of Tech Support are major risk factors. Try changing the Contract Offer to One or Two Year, and enable Tech Support / Online Security to force a significant risk drop.")
            st.markdown("</div>", unsafe_allow_html=True)
