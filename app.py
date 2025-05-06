import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import calendar
from datetime import datetime, timedelta
import random

# for seaborn/matplotlib charts later
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Enhanced HR Planning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 1.5rem;
    }
    .card {
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        background-color: #FFFFFF;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
        font-weight: 500;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
    }
    .trend-positive {
        color: #10B981;
        font-size: 0.9rem;
    }
    .trend-negative {
        color: #EF4444;
        font-size: 0.9rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .plotly-chart {
        width: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Function to generate sample data
@st.cache_data
def generate_sample_data(n=500):
    np.random.seed(42)
    
    departments = [
        "Engineering", "Marketing", "Sales", "Finance", 
        "Human Resources", "Operations", "IT", "Customer Support",
        "Research & Development", "Product Management"
    ]
    
    job_roles = {
        "Engineering": ["Software Engineer", "DevOps Engineer", "QA Engineer", "Engineering Manager"],
        "Marketing": ["Marketing Specialist", "Content Writer", "SEO Specialist", "Marketing Manager"],
        "Sales": ["Sales Representative", "Account Executive", "Sales Manager", "Business Development"],
        "Finance": ["Financial Analyst", "Accountant", "Finance Manager", "Controller"],
        "Human Resources": ["HR Coordinator", "Recruiter", "HR Manager", "Talent Acquisition"],
        "Operations": ["Operations Analyst", "Project Manager", "Operations Director", "Process Improvement"],
        "IT": ["Systems Administrator", "IT Support", "IT Manager", "Network Engineer"],
        "Customer Support": ["Support Agent", "Customer Success", "Support Manager", "Customer Experience"],
        "Research & Development": ["Research Scientist", "Product Developer", "R&D Manager", "Innovation Lead"],
        "Product Management": ["Product Manager", "Product Owner", "UX Designer", "Product Director"]
    }
    
    locations = ["New York", "San Francisco", "Chicago", "Austin", "Denver", "Seattle", "Los Angeles", "Boston", "Atlanta", "Remote"]
    
    genders = ["Male", "Female", "Non-Binary"]
    gender_weights = [0.48, 0.47, 0.05]
    
    # Generate random data
    data = {
        "EmployeeID": list(range(1001, 1001 + n)),
        "Name": [f"Employee_{i}" for i in range(n)],
        "Age": np.random.randint(22, 65, n),
        "Department": np.random.choice(departments, n),
        "Salary": np.random.randint(40000, 200000, n),
        "Tenure": np.random.randint(0, 20, n),
        "Gender": np.random.choice(genders, n, p=gender_weights),
        "Location": np.random.choice(locations, n),
        "Engagement": np.clip(np.random.normal(3.5, 0.8, n), 1, 5),
        "Satisfaction": np.clip(np.random.normal(3.7, 0.9, n), 1, 5),
        "Performance": np.clip(np.random.normal(3.6, 0.7, n), 1, 5),
        "RetirementRisk": np.zeros(n),
        "LeadershipPotential": np.clip(np.random.normal(3.4, 0.8, n), 1, 5),
        "ManagerRating": np.clip(np.random.normal(3.8, 0.6, n), 1, 5),
        "WorkLifeBalance": np.clip(np.random.normal(3.2, 0.9, n), 1, 5),
        "TrainingHours": np.random.randint(0, 80, n),
        "PromotionEligible": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "HiringDate": [
    (datetime.now() - timedelta(days=int(365 * t + random.randint(0, 364)))).strftime('%Y-%m-%d')
    for t in np.random.randint(0, 20, n)
],
    }
    
    # Add job roles based on department
    data["JobRole"] = [np.random.choice(job_roles[dept]) for dept in data["Department"]]
    
    # Calculate retirement risk based on age
    for i in range(n):
        if data["Age"][i] >= 60:
            data["RetirementRisk"][i] = np.random.uniform(0.6, 0.9)
        elif data["Age"][i] >= 55:
            data["RetirementRisk"][i] = np.random.uniform(0.3, 0.6)
        elif data["Age"][i] >= 50:
            data["RetirementRisk"][i] = np.random.uniform(0.1, 0.3)
    
    # Create age groups
    age_bins = [20, 30, 40, 50, 60, 70]
    age_labels = ['20-29', '30-39', '40-49', '50-59', '60+']
    data["AgeGroup"] = pd.cut(data["Age"], bins=age_bins, labels=age_labels, right=False)
    
    # Add attrition data (historical and predicted)
    attrition_prob = np.zeros(n)
    for i in range(n):
        base_prob = 0.12  # Base attrition probability
        
        # Factors that increase attrition risk
        if data["Satisfaction"][i] < 2.5:
            base_prob += 0.15
        if data["Engagement"][i] < 2.5:
            base_prob += 0.12
        if data["Performance"][i] < 2.8:
            base_prob += 0.05
        if data["WorkLifeBalance"][i] < 2.5:
            base_prob += 0.10
        if data["ManagerRating"][i] < 2.5:
            base_prob += 0.13
        if data["Tenure"][i] < 2:
            base_prob += 0.08
        
        # Factors that decrease attrition risk
        if data["Tenure"][i] > 10:
            base_prob -= 0.10
        if data["Age"][i] > 50:
            base_prob -= 0.08
        if data["PromotionEligible"][i] == 1:
            base_prob -= 0.07
        
        # Cap probability
        attrition_prob[i] = max(0.01, min(0.95, base_prob))
    
    data["AttritionProbability"] = attrition_prob
    data["Attrition"] = np.random.binomial(1, attrition_prob)
    
    data["JobInvolvement"]          = np.clip(np.random.normal(3.3, 0.8, n), 1, 5)
    data["EnvironmentSatisfaction"] = np.clip(np.random.normal(3.4, 0.8, n), 1, 5)
    
    
    
    # Generate monthly attrition data for the past year
    months = [calendar.month_name[i] for i in range(1, 13)]
    attrition_trend = {}
    for dept in departments:
        attrition_trend[dept] = [round(max(0.02, min(0.25, np.random.normal(0.12, 0.04))), 3) for _ in range(12)]
    
    # Market data - average salaries by role
    market_data = {}
    for dept, roles in job_roles.items():
        market_data[dept] = {role: int(np.random.normal(100000, 30000)) for role in roles}
    
    # Success plans
    succession_data = []
    critical_roles = ["Engineering Manager", "Finance Manager", "Sales Manager", "Marketing Manager", 
                     "HR Manager", "IT Manager", "Operations Director", "Support Manager", 
                     "R&D Manager", "Product Director"]
    
    for i in range(n):
        if data["JobRole"][i] in critical_roles:
            # Find potential successors
            dept = data["Department"][i]
            candidates = []
            for j in range(n):
                if (data["Department"][j] == dept and 
                    data["LeadershipPotential"][j] >= 4.0 and 
                    data["Performance"][j] >= 4.0 and
                    data["Tenure"][j] >= 3):
                    candidates.append(j)
            
            if candidates:
                # Select up to 3 successors
                successors = random.sample(candidates, min(3, len(candidates)))
                succession_data.append({
                    "CriticalRoleID": data["EmployeeID"][i],
                    "CriticalRole": data["JobRole"][i],
                    "Department": dept,
                    "CurrentHolder": data["Name"][i],
                    "Successors": [data["Name"][s] for s in successors],
                    "SuccessorIDs": [data["EmployeeID"][s] for s in successors],
                    "ReadinessLevels": [random.choice(["Ready Now", "Ready in 1-2 Years", "Ready in 3+ Years"]) for _ in successors]
                })
    
    df = pd.DataFrame(data)
    
    return df, attrition_trend, market_data, succession_data

# Load or generate data
# — ALWAYS USE SAMPLE DATA FOR DEMO —
df, attrition_trend, market_data, succession_data = generate_sample_data(600)


# Sidebar - filters and navigation
with st.sidebar:
    st.image("sabadell.jpg", width=150)
    st.markdown("### Filters")
    
    # Date range filter
    today = datetime.now()
    start_date = st.date_input(
        "Start Date",
        datetime(today.year - 1, today.month, 1)
    )
    end_date = st.date_input(
        "End Date",
        today
    )
    
    # Department filter with checkboxes
    st.markdown("#### Department")
    all_depts = st.checkbox("All Departments", value=True)
    if all_depts:
        departments = df['Department'].unique()
    else:
        departments = st.multiselect(
            "Select Departments",
            options=df['Department'].unique(),
            default=df['Department'].unique()[:3]
        )
    
    # Age group filter
    st.markdown("#### Age Group")
    all_ages = st.checkbox("All Age Groups", value=True)
    if all_ages:
        age_groups = df['AgeGroup'].unique()
    else:
        age_groups = st.multiselect(
            "Select Age Groups",
            options=df['AgeGroup'].unique(),
            default=df['AgeGroup'].unique()
        )
    
    # Gender filter
    st.markdown("#### Gender")
    all_genders = st.checkbox("All Genders", value=True)
    if all_genders:
        genders = df['Gender'].unique()
    else:
        genders = st.multiselect(
            "Select Genders",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )
    
    # Location filter
    st.markdown("#### Location")
    all_locations = st.checkbox("All Locations", value=True)
    if all_locations:
        locations = df['Location'].unique()
    else:
        locations = st.multiselect(
            "Select Locations",
            options=df['Location'].unique(),
            default=df['Location'].unique()[:3]
        )

# Apply filters
filtered_df = df[
    (df['Department'].isin(departments)) &
    (df['AgeGroup'].isin(age_groups)) &
    (df['Gender'].isin(genders)) &
    (df['Location'].isin(locations))
]

# Main content
st.markdown('<h1 class="main-header">Enhanced HR Planning Dashboard</h1>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏠 Overview", 
    "🔁 Attrition", 
    "😊 Engagement", 
    "👵 Retirement", 
    "🌐 Market Analysis", 
    "🔄 Succession Plan",
    "⚙️ Settings"
])

# -------------------------
# Tab 1: Overview
# -------------------------
with tab1:
    st.markdown('<h2 class="section-header">Executive Overview</h2>', unsafe_allow_html=True)
    
    # Create four metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    
    # Metric 1: Total Employees
    with col1:
        st.markdown("""
        <div class="card">
            <div class="metric-label">Total Employees</div>
            <div class="metric-value">{}</div>
            <div class="trend-positive">↑ 5.2% from last quarter</div>
        </div>
        """.format(len(filtered_df)), unsafe_allow_html=True)
    
    # Metric 2: Avg Engagement
    engagement_avg = round(filtered_df['Engagement'].mean(), 2)
    engagement_trend = 0.3  # Demo value
    trend_class = "trend-positive" if engagement_trend >= 0 else "trend-negative"
    trend_symbol = "↑" if engagement_trend >= 0 else "↓"
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Avg Engagement</div>
            <div class="metric-value">{engagement_avg}</div>
            <div class="{trend_class}">{trend_symbol} {abs(engagement_trend)} from last quarter</div>
        </div>
        """, unsafe_allow_html=True)
    
       # Metric 3: Attrition Rate
    attrition_rate = round(filtered_df['Attrition'].mean() * 100, 1)
    attrition_trend_kpi = -1.2  # Demo value
    trend_class   = "trend-positive" if attrition_trend_kpi <= 0 else "trend-negative"
    trend_symbol  = "↓" if attrition_trend_kpi <= 0 else "↑"
    
    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Attrition Rate</div>
            <div class="metric-value">{attrition_rate}%</div>
            <div class="{trend_class}">{trend_symbol} {abs(attrition_trend_kpi)}% from last quarter</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Metric 4: Avg Satisfaction
    satisfaction_avg = round(filtered_df['Satisfaction'].mean(), 2)
    satisfaction_trend = 0.2  # Demo value
    trend_class = "trend-positive" if satisfaction_trend >= 0 else "trend-negative"
    trend_symbol = "↑" if satisfaction_trend >= 0 else "↓"
    
    with col4:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Avg Satisfaction</div>
            <div class="metric-value">{satisfaction_avg}</div>
            <div class="{trend_class}">{trend_symbol} {abs(satisfaction_trend)} from last quarter</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Create two charts in a row
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Department Distribution</h3>', unsafe_allow_html=True)
        dept_counts = filtered_df['Department'].value_counts().reset_index()
        dept_counts.columns = ['Department', 'Count']
        
        fig = px.bar(
            dept_counts, 
            x='Department', 
            y='Count',
            title='Employee Count by Department',
            color='Department',
            text='Count',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Number of Employees")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True, key="overview_dept")
    
    with col2:
        st.markdown('<h3 class="section-header">Gender Distribution</h3>', unsafe_allow_html=True)
        gender_counts = filtered_df['Gender'].value_counts().reset_index()
        gender_counts.columns = ['Gender', 'Count']
        
        fig = px.pie(
            gender_counts, 
            values='Count', 
            names='Gender',
            title='Gender Distribution',
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.4
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="overview_gender")
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Age Distribution</h3>', unsafe_allow_html=True)
        age_counts = filtered_df['AgeGroup'].value_counts().reset_index()
        age_counts.columns = ['AgeGroup', 'Count']
        age_counts = age_counts.sort_values('AgeGroup')
        
        fig = px.bar(
            age_counts,
            x='AgeGroup',
            y='Count',
            title='Employee Count by Age Group',
            color='AgeGroup',
            text='Count',
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=400, xaxis_title="Age Group", yaxis_title="Number of Employees")
        st.plotly_chart(fig, use_container_width=True, key="overview_age")
    
    with col2:
        st.markdown('<h3 class="section-header">Performance vs Engagement</h3>', unsafe_allow_html=True)
        
        fig = px.scatter(
            filtered_df,
            x='Performance',
            y='Engagement',
            color='Department',
            title='Performance vs Engagement',
            opacity=0.7,
            template='plotly_white',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="overview_scatter")

# -------------------------
# Tab 2: Attrition Analysis
# -------------------------
with tab2:
    st.markdown('<h2 class="section-header">Attrition Analysis</h2>', unsafe_allow_html=True)
    
    # Main metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_attrition = round(filtered_df['Attrition'].mean() * 100, 1)
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Current Attrition Rate</div>
            <div class="metric-value">{current_attrition}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_risk = len(filtered_df[filtered_df['AttritionProbability'] > 0.4])
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">High Risk Employees</div>
            <div class="metric-value">{high_risk}</div>
            <div class="trend-negative">Attrition probability > 40%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        key_talent_risk = len(filtered_df[(filtered_df['AttritionProbability'] > 0.4) & 
                                        (filtered_df['Performance'] > 4.0)])
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">Key Talent at Risk</div>
            <div class="metric-value">{key_talent_risk}</div>
            <div class="trend-negative">High performers at risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h3 class="section-header">Attrition by Department</h3>', unsafe_allow_html=True)
        dept_attrition = filtered_df.groupby('Department')['Attrition'].mean().reset_index()
        dept_attrition['Attrition'] = dept_attrition['Attrition'] * 100
        dept_attrition = dept_attrition.sort_values('Attrition', ascending=False)
        
        fig = px.bar(
            dept_attrition,
            x='Department',
            y='Attrition',
            title='Attrition Rate by Department (%)',
            color='Attrition',
            text_auto='.1f',
            color_continuous_scale='Reds',
            template='plotly_white'
        )
        fig.update_layout(height=400, xaxis_title="", yaxis_title="Attrition Rate (%)")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True, key="attr_dept")
    
    with col2:
        st.markdown('<h3 class="section-header">Attrition by Age Group</h3>', unsafe_allow_html=True)
        age_attrition = filtered_df.groupby('AgeGroup')['Attrition'].mean().reset_index()
        age_attrition['Attrition'] = age_attrition['Attrition'] * 100
        age_attrition = age_attrition.sort_values('AgeGroup')
        
        fig = px.bar(
            age_attrition,
            x='AgeGroup',
            y='Attrition',
            title='Attrition Rate by Age Group (%)',
            color='Attrition',
            text_auto='.1f',
            color_continuous_scale='Reds',
            template='plotly_white'
        )
        fig.update_layout(height=400, xaxis_title="Age Group", yaxis_title="Attrition Rate (%)")
        st.plotly_chart(fig, use_container_width=True, key="attr_agegrp")
    
            # Historical attrition trends
        st.markdown('<h3 class="section-header">Historical Attrition Trends</h3>', unsafe_allow_html=True)

        # Generate months & select departments
        months = [calendar.month_name[i] for i in range(1, 13)]
        selected_depts = st.multiselect(
            "Select departments to display",
            options=departments,
            default=departments[:3]
        )

        # Build the figure
        fig = go.Figure()

        # Add one trace per department
        for dept in selected_depts:
            rates = attrition_trend.get(dept, [])
            if not rates:
                st.warning(f"No trend data for {dept}")
                continue

            fig.add_trace(go.Scatter(
                x=months,
                y=[round(r * 100, 1) for r in rates],
                mode='lines+markers',
                name=dept
            ))

        # ─── Dedented to this level ─────────────────────────────
        fig.update_layout(
            title='Monthly Attrition Rate by Department (%)',
            xaxis_title='Month',
            yaxis_title='Attrition Rate (%)',
            template='plotly_white',
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, key="tab2_attr_trends")

    
    # Risk factors
    st.markdown('<h3 class="section-header">Attrition Risk Factors</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation between satisfaction and attrition probability
        fig = px.scatter(
            filtered_df,
            x='Satisfaction',
            y='AttritionProbability',
            color='Department',
            title='Satisfaction vs Attrition Risk',
            opacity=0.7,
            template='plotly_white',
            trendline='ols',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=400, xaxis_title="Satisfaction Score", yaxis_title="Attrition Probability")
        st.plotly_chart(fig, use_container_width=True, key="tab2_risk_satisfaction")
    
    with col2:
        # Correlation between tenure and attrition probability
        fig = px.scatter(
            filtered_df,
            x='Tenure',
            y='AttritionProbability',
            color='Department',
            title='Tenure vs Attrition Risk',
            opacity=0.7,
            template='plotly_white',
            trendline='ols',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(height=400, xaxis_title="Tenure (Years)", yaxis_title="Attrition Probability")
        st.plotly_chart(fig, use_container_width=True, key="tab2_risk_tenure")

# -------------------------
# --- Engagement & Satisfaction with tab3:
with tab3:
    st.subheader("😊 Engagement & Satisfaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Satisfaction by Department")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Department', y='Satisfaction', data=filtered_df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Engagement by Department")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Department', y='Engagement', data=filtered_df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    st.markdown("### Engagement vs Satisfaction")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Satisfaction', y='Engagement', hue='Department', 
                   size='Performance', sizes=(50, 200), data=filtered_df, ax=ax)
    plt.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
    
    st.markdown("### Factors Affecting Engagement")
    engagement_cols = ['WorkLifeBalance', 'JobInvolvement', 'EnvironmentSatisfaction']
    engagement_data = filtered_df[engagement_cols + ['Department']].melt(
        id_vars=['Department'], var_name='Factor', value_name='Score')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Factor', y='Score', hue='Department', data=engagement_data, ax=ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# --- Retirement & Demographics with tab4:
with tab4:
    st.subheader("👵 Retirement & Demographics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Age Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df['Age'], bins=15, kde=True, ax=ax)
        plt.axvline(x=55, color='red', linestyle='--', label='Retirement Risk Age')
        plt.legend()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Retirement Risk by Department")
        dept_retirement = filtered_df.groupby('Department')['RetirementRisk'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=dept_retirement.index, y=dept_retirement.values, ax=ax)
        plt.xticks(rotation=45)
        plt.ylabel('Average Retirement Risk Score')
        st.pyplot(fig)
    
    st.markdown("### Employees Near Retirement")
    retirement_risk_df = filtered_df[filtered_df['RetirementRisk'] > 0.5]
    st.dataframe(retirement_risk_df[['Department', 'JobRole', 'Age', 'Tenure', 'RetirementRisk']])
    
    st.markdown("### Age Group Distribution by Department")
    age_dept_counts = pd.crosstab(filtered_df['Department'], filtered_df['AgeGroup'])
    fig, ax = plt.subplots(figsize=(12, 8))
    age_dept_counts.plot(kind='bar', stacked=True, ax=ax)
    plt.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel('Number of Employees')
    st.pyplot(fig)

# --- Market Role Intelligence with tab5:
with tab5:
    st.subheader("🌐 External Role Demand")
    
    st.info("This section displays external job market trends and role demand.")
    
    # Placeholder tabs for external data sources
    market_tab1, market_tab2, market_tab3 = st.tabs(["Salary Trends", "Skills Demand", "Industry Growth"])
    
    with market_tab1:
        st.markdown("### Salary Benchmarks by Role")
        # Placeholder for salary data visualization
        salary_data = {
            'Data Scientist': [85000, 95000, 120000],
            'Software Engineer': [80000, 90000, 115000],
            'Product Manager': [90000, 105000, 130000],
            'Marketing Specialist': [65000, 75000, 90000],
            'HR Manager': [70000, 85000, 100000]
        }
        
        salary_df = pd.DataFrame(salary_data, index=['25th Percentile', 'Median', '75th Percentile']).T
        salary_df = salary_df.reset_index().rename(columns={'index': 'Role'})
        salary_melted = salary_df.melt(id_vars=['Role'], var_name='Percentile', value_name='Salary')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Role', y='Salary', hue='Percentile', data=salary_melted, ax=ax)
        plt.xticks(rotation=45)
        plt.title('Market Salary Benchmarks')
        st.pyplot(fig)
        
        st.markdown("### Internal vs Market Comparison")
        st.write("Upload your market salary data to compare with internal compensation.")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    with market_tab2:
        st.markdown("### In-Demand Skills by Department")
        skills_data = {
            'Technical': ['Data Analysis', 'Cloud Computing', 'AI/ML', 'Cybersecurity', 'DevOps'],
            'Management': ['Leadership', 'Strategic Planning', 'Change Management', 'Performance Management'],
            'Soft Skills': ['Communication', 'Problem Solving', 'Adaptability', 'Collaboration']
        }
        
        for category, skills in skills_data.items():
            st.write(f"**{category}:**")
            for i, skill in enumerate(skills):
                st.write(f"- {skill}")
            st.write("")
    
    with market_tab3:
        st.markdown("### Industry Growth Projections")
        growth_data = {
            'Industry': ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail'],
            'Growth Rate': [12, 15, 8, 4, 3]
        }
        growth_df = pd.DataFrame(growth_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Industry', y='Growth Rate', data=growth_df, ax=ax)
        plt.ylabel('Projected Growth Rate (%)')
        plt.title('5-Year Industry Growth Projections')
        st.pyplot(fig)

# --- Succession Planning with tab6:
with tab6:
    st.subheader("🔄 Succession Planning Tool")
    
    st.markdown("### Critical Roles at Risk")
    
    # Filter for high-risk roles
    high_risk_roles = filtered_df[(filtered_df['RetirementRisk'] > 0.5) | 
                                (filtered_df['Attrition'] > 0.3)]
    
    if not high_risk_roles.empty:
        st.dataframe(high_risk_roles[['EmployeeID', 'JobRole', 'Department', 
                                    'RetirementRisk', 'Attrition', 'Performance']])
    else:
        st.info("No critical roles at high risk based on current filters")
    
    st.markdown("### Succession Readiness Matrix")
    
    # Create a sample succession readiness matrix
    succession_data = {
        'Critical Role': ['Director of Technology', 'Senior Data Scientist', 'VP of Sales', 'HR Manager', 'Finance Director'],
        'Current Holder': ['Employee 101', 'Employee 245', 'Employee 052', 'Employee 198', 'Employee 073'],
        'Risk Level': ['High', 'Medium', 'High', 'Low', 'Medium'],
        'Successor 1': ['Employee 153', 'Employee 301', 'Employee 128', 'Employee 222', 'Employee 095'],
        'Successor 1 Readiness': [0.8, 0.9, 0.7, 0.6, 0.5],
        'Successor 2': ['Employee 178', 'Employee 316', 'Employee 145', 'Employee 235', 'Employee 112'],
        'Successor 2 Readiness': [0.7, 0.7, 0.5, 0.5, 0.4]
    }
    
    succession_df = pd.DataFrame(succession_data)
    
    # Display succession planning matrix
    st.dataframe(succession_df)
    
    # Visualization of successor readiness
    st.markdown("### Successor Readiness Levels")
    
    successor_data = pd.DataFrame({
        'Role': succession_df['Critical Role'],
        'Successor 1': succession_df['Successor 1 Readiness'],
        'Successor 2': succession_df['Successor 2 Readiness']
    })
    
    successor_melted = successor_data.melt(id_vars=['Role'], var_name='Successor', value_name='Readiness')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Role', y='Readiness', hue='Successor', data=successor_melted, ax=ax)
    plt.xticks(rotation=45)
    plt.title('Successor Readiness by Critical Role')
    plt.ylim(0, 1)
    st.pyplot(fig)
    
    # Add a form to update succession plans
    st.markdown("### Update Succession Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_role = st.selectbox("Select Critical Role", succession_df['Critical Role'])
        risk_level = st.selectbox("Risk Level", ["Low", "Medium", "High"])
    
    with col2:
        successor_1 = st.text_input("Successor 1")
        successor_1_readiness = st.slider("Successor 1 Readiness", 0.0, 1.0, 0.5, 0.1)
    
    if st.button("Update Succession Plan"):
        st.success(f"Succession plan updated for {selected_role}")

# --- Settings with tab7:
with tab7:
    st.subheader("⚙️ Settings & Data Management")
    
    settings_tab1, settings_tab2, settings_tab3 = st.tabs(["Data Preview", "Dashboard Settings", "Export Options"])
    
    with settings_tab1:
        st.markdown("### Dataset Overview")
        
        st.dataframe(filtered_df.head(10))
        
        st.markdown("### Data Summary")
        st.write(f"**Total Records:** {len(filtered_df)}")
        st.write(f"**Departments:** {', '.join(filtered_df['Department'].unique())}")
        st.write(f"**Job Roles:** {', '.join(filtered_df['JobRole'].unique())}")
        
        # Data quality checks
        st.markdown("### Data Quality Checks")
        
        missing_values = filtered_df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        
        if not missing_values.empty:
            st.warning("Missing values detected in the dataset:")
            st.write(missing_values)
        else:
            st.success("No missing values in the dataset")
    
    with settings_tab2:
        st.markdown("### Dashboard Configuration")
        
        st.markdown("#### Chart Settings")
        chart_theme = st.selectbox("Chart Theme", ["default", "darkgrid", "whitegrid", "dark", "white", "ticks"])
        chart_palette = st.selectbox("Color Palette", ["viridis", "plasma", "inferno", "magma", "cividis", "muted", "pastel"])
        
        # Apply theme settings
        if st.button("Apply Chart Settings"):
            sns.set_theme(style=chart_theme)
            sns.set_palette(chart_palette)
            st.success("Chart settings updated")
        
        st.markdown("#### Data Refresh Settings")
        refresh_frequency = st.radio("Data Refresh Frequency", ["Manual", "Daily", "Weekly", "Monthly"])
        
        if refresh_frequency != "Manual":
            st.info(f"Data will be automatically refreshed {refresh_frequency.lower()}")
        
        if st.button("Refresh Data Now"):
            st.success("Data refreshed successfully")
            st.cache_data.clear()
    
    with settings_tab3:
        st.markdown("### Export Options")
        
        export_format = st.radio("Export Format", ["CSV", "Excel", "JSON"])
        export_data = st.multiselect("Select Data to Export", 
                                  ["Employee Data", "Department Analytics", "Engagement Metrics", 
                                   "Retirement Risk", "Succession Plans"])
        
        if st.button("Generate Export"):
            if export_data:
                st.success(f"Export generated in {export_format} format")
                st.download_button(
                    label=f"Download {export_format} File",
                    data="Sample export data",
                    file_name=f"hr_dashboard_export.{export_format.lower()}",
                    mime="text/plain"
                )
            else:
                st.error("Please select at least one data category to export")