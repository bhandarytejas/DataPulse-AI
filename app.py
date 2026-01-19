"""
DataPulse AI - Automated Data Quality Monitor with AI Explainations
Built with Streamlit and Google Gemini
Updated: Jan 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from google import genai
from google.genai import types
from datetime import datetime
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')


# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="DataPulse AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# INITIALIZE GOOGLE GEMINI (NEW SDK)
# ============================================
@st.cache_resource
def init_gemini():
    """
    Initializing the Google Gemini client with the API key.
    """
    api_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

    if not api_key:
        st.error("""
                 ⚠️ **Gemini API Key Required**
        
                Please add your API key in one of these ways:
                
                **For local development:**
                Create `.streamlit/secrets.toml` with: GEMINI_API_KEY = "your-api-key-here"
                        
                **For Streamlit Cloud:**
                Add to your app's Secrets in the dashboard.
                
                **Get a free API key at:** https://aistudio.google.com/app/apikey
                """)
        st.stop()

    # Initialize the NEW client
    try: 
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Gemini Client: {str(e)}")
        st.stop()    

# Initialize client globally
gemini_client = init_gemini()

# ============================================
# AI EXPLANATION FUNCTION
# ============================================
def get_ai_explanation(issue_description: str, data_context: str) -> str:
    """
    Get AI explanation for a data quality issue using Google Gemini.

    Args:
        issue_description (str): Description of the data quality issue.
        data_context (str): Contextual information about the dataset.

    Returns:
        str: AI-generated explanation.
    """
    prompt = f"""You are a senior data analyst explaining a data quality issue to a business stakeholder.

        ##Issue Description: {issue_description}
        ##Data Context: {data_context}

        ## Your Task:
        Provide a clear, actionable explanation in 3-4 sentences covering:
        1. What this issue means in practical business terms
        2. The potential impact or consequences if not addressed
        3. The most likely cause or root of the problem
        4. A specific, actionable recommendation to fix it

        ## Guidelines:
        - Be direct and practical
        - Avoid technical jargon - explain for non-technical stakeholders
        - Focus on business impact, not just technical details
        - Provide ONE specific action item
        """
    try:
        response = gemini_client.models.generate_content(
            model = "gemini-2.5-flash",
            contents=prompt,
            #max_output_tokens=500
        )
        return response.text
    except Exception as e:
        return f"Could not generate AI explanation: {str(e)}. Please check your API key and rate limits."

# ============================================
# DATA QUALITY DETECTION FUNCTIONS
# ============================================
def detect_missing_values(df: pd.DataFrame) -> list:
    """Detect missing values and assess their impact."""
    issues = []
    missing_counts = df.isnull().sum()

    for col in missing_counts[missing_counts > 0].index:
        count = missing_counts[col]
        percentage = (count / len(df)) * 100

        # Determin severity based on percentage
        if percentage  > 20:
            severity = "🔴 Critical"
        elif percentage  > 10:
            severity = "🟠 High"
        elif percentage  > 5:
            severity = "🟡 Medium"
        else:
            severity = "🟢 Low"

        # Build detailed issue description
        issue_desc = f"""
        Column: '{col}
        Missing Count: {count} out of {len(df)} total rows ({percentage:.1f}%)
        Data Type: {df[col].dtype}
        """
        # Build data context
        if df[col].dtype in [np.float64, np.int64]:
            non_null_status = df[col].dropna().describe()
            data_context = f"""
            This is a numeric column.
            When values exis, the range is {non_null_status['min']:.2f} to {non_null_status['max']:.2f}, 
            with a mean of {non_null_status['mean']:.2f}.
            """
        else:
            unique_count = df[col].dropna().unique()
            data_context = f"""
            This is a text/categorical column with {unique_count} unique values.
            Dataset has {len(df)} total rows.
            """

    issues.append({
                'type': 'Missing Values',
                'column': col,
                'severity': severity,
                'count': count,
                'percentage': percentage,
                'description': f"{count:,} missing values ({percentage:.1f}%)",
                'ai_explanation': get_ai_explanation(issue_desc, data_context)
            })
    
    return issues

def detect_outliers(df: pd.DataFrame) -> list:
    """Detect outliers using IQR method"""
    issues = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        valid_data = df[col].dropna()
        if len(valid_data) < 10:
            continue
        
        # Calculate IQR
        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds (1.5× IQR is standard)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Find outliers
        outliers_mask = (valid_data < lower_bound) | (valid_data > upper_bound)
        num_outliers = outliers_mask.sum()
        
        if num_outliers > 0:
            outlier_values = valid_data[outliers_mask].values
            outlier_percentage = (num_outliers / len(valid_data)) * 100
            
            # Severity
            if outlier_percentage > 5:
                severity = "🟠 High"
            elif outlier_percentage > 2:
                severity = "🟡 Medium"
            else:
                severity = "🟢 Low"
            
            issue_desc = f"""
                    Column: '{col}'
                    Outliers Found: {num_outliers} values ({outlier_percentage:.1f}%)
                    Method: IQR (values outside Q1-1.5×IQR to Q3+1.5×IQR)

                    Statistics:
                    - Q1 (25th percentile): {q1:.2f}
                    - Q3 (75th percentile): {q3:.2f}
                    - IQR: {iqr:.2f}
                    - Valid range: {lower_bound:.2f} to {upper_bound:.2f}

                    Example outliers: {sorted(outlier_values[:5].tolist())}
                    """
            
            data_context = f"""
                    This numeric column has a middle 50% range of {q1:.2f} to {q3:.2f}.
                    Values below {lower_bound:.2f} or above {upper_bound:.2f} are flagged.
                    IQR method is robust and doesn't assume normal distribution.
                    """
            
            issues.append({
                'type': 'Outliers',
                'column': col,
                'severity': severity,
                'count': num_outliers,
                'percentage': outlier_percentage,
                'description': f"{num_outliers:,} outliers detected ({outlier_percentage:.1f}%)",
                'details': f"Valid range: {lower_bound:.2f} to {upper_bound:.2f}",
                'outlier_values': sorted(outlier_values[:10].tolist()),
                'ai_explanation': get_ai_explanation(issue_desc, data_context)
            })
    
    return issues


def detect_duplicates(df: pd.DataFrame) -> list:
    """Detect duplicate rows."""
    issues = []
    
    # Check for exact duplicates
    num_duplicates = df.duplicated().sum()
    
    if num_duplicates > 0:
        percentage = (num_duplicates / len(df)) * 100
        
        # Determine severity
        if percentage > 10:
            severity = "🔴 Critical"
        elif percentage > 5:
            severity = "🟠 High"
        else:
            severity = "🟡 Medium"
        
        # Find which rows are duplicated
        duplicate_rows = df[df.duplicated(keep='first')]
        
        issue_desc = f"""
                    Duplicate Rows Found: {num_duplicates} out of {len(df)} total rows ({percentage:.1f}%)
                    These are EXACT duplicates across all {len(df.columns)} columns.
                    """
        
        data_context = f"""
                    Dataset has {len(df)} rows and {len(df.columns)} columns.
                    {percentage:.1f}% of rows are exact duplicates of earlier rows.
                    This could indicate data loading issues, ETL problems, or intentional redundancy.
                    """
        
        issues.append({
            'type': 'Duplicate Rows',
            'column': 'All columns',
            'severity': severity,
            'count': num_duplicates,
            'percentage': percentage,
            'description': f"{num_duplicates:,} duplicate rows ({percentage:.1f}%)",
            'ai_explanation': get_ai_explanation(issue_desc, data_context)
        })
    
    return issues


def detect_data_type_issues(df: pd.DataFrame) -> list:
    """Detect columns that might have incorrect data types."""
    issues = []
    
    for col in df.columns:
        # Check if object column could be numeric
        if df[col].dtype == 'object':
            try:
                # Try to convert to numeric
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                non_null_count = numeric_vals.notna().sum()
                total_non_null = df[col].notna().sum()
                
                if total_non_null > 0:
                    conversion_rate = non_null_count / total_non_null
                    
                    # If >80% can be numeric, flag it
                    if conversion_rate > 0.8:
                        percentage = conversion_rate * 100
                        
                        issue_desc = f"""
                                Column: '{col}'
                                Current Type: Text/String
                                Issue: {percentage:.0f}% of values can be converted to numbers

                                This column is stored as text but appears to contain mostly numeric data.
                                This prevents mathematical operations and may cause analysis errors.
                                """
                                                        
                        data_context = f"""
                                Sample values from this column: {df[col].dropna().head(5).tolist()}
                                The column should likely be numeric for proper analysis.
                                """
                        
                        issues.append({
                            'type': 'Data Type Issue',
                            'column': col,
                            'severity': '🟡 Medium',
                            'count': int(non_null_count),
                            'percentage': percentage,
                            'description': f"Stored as text but {percentage:.0f}% are numbers",
                            'ai_explanation': get_ai_explanation(issue_desc, data_context)
                        })
            except Exception:
                pass
        
        # Check if column could be datetime
        if df[col].dtype == 'object' and 'date' in col.lower():
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                valid_count = parsed.notna().sum()
                total_count = df[col].notna().sum()
                
                if total_count > 0 and valid_count / total_count > 0.8:
                    issue_desc = f"""
                        Column: '{col}'
                        Current Type: Text/String
                        Issue: Column name suggests dates but stored as text

                        {valid_count} values can be parsed as dates.
                        Storing dates as text prevents date-based filtering, sorting, and calculations.
                        """
                    
                    issues.append({
                        'type': 'Data Type Issue',
                        'column': col,
                        'severity': '🟢 Low',
                        'count': valid_count,
                        'percentage': (valid_count / total_count) * 100,
                        'description': f"Appears to be dates stored as text",
                        'ai_explanation': get_ai_explanation(issue_desc, "Date column stored as string")
                    })
            except Exception:
                pass
    
    return issues


def detect_inconsistent_formats(df: pd.DataFrame) -> list:
    """Detect inconsistent formatting in text columns."""
    issues = []
    text_cols = df.select_dtypes(include=['object']).columns
    
    for col in text_cols:
        if df[col].dtype != 'object':
            continue
            
        sample_values = df[col].dropna()
        if len(sample_values) < 10:
            continue
        
        # Check for case inconsistencies
        lower_values = sample_values.str.lower()
        unique_original = sample_values.nunique()
        unique_lower = lower_values.nunique()
        
        inconsistent_count = unique_original - unique_lower
        
        if inconsistent_count > 2:  # More than 2 case variations
            # Find examples
            value_counts_original = sample_values.value_counts()
            value_counts_lower = lower_values.value_counts()
            
            issue_desc = f"""
                    Column: '{col}'
                    Issue: {inconsistent_count} values differ only in capitalization

                    For example, 'New York', 'new york', and 'NEW YORK' would be treated as 
                    different categories, creating artificial duplicates in your analysis.

                    Original unique values: {unique_original}
                    After normalizing case: {unique_lower}
                    """
                                
            data_context = f"""
                    This is a text column with categorical data.
                    Inconsistent capitalization can cause:
                    - Incorrect groupby results
                    - Misleading counts and percentages  
                    - Issues with joins and lookups
                    """
            
            issues.append({
                'type': 'Format Inconsistency',
                'column': col,
                'severity': '🟢 Low',
                'count': inconsistent_count,
                'percentage': 0,
                'description': f"{inconsistent_count} case variations found",
                'ai_explanation': get_ai_explanation(issue_desc, data_context)
            })
    
    return issues


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def create_issue_summary_chart(issues: list) -> go.Figure:
    """Create a summary chart of issues by type."""
    if not issues:
        return None
    
    df_issues = pd.DataFrame(issues)
    type_counts = df_issues.groupby('type').size().reset_index(name='count')
    
    fig = px.bar(
        type_counts,
        x='type',
        y='count',
        title="Issues by Type",
        labels={'type': 'Issue Type', 'count': 'Number of Issues'},
        color='count',
        color_continuous_scale='Reds'
    )
    fig.update_layout(showlegend=False)
    return fig


def create_severity_chart(issues: list) -> go.Figure:
    """Create a pie chart of issues by severity."""
    if not issues:
        return None
    
    severity_counts = pd.DataFrame(issues)['severity'].value_counts()
    
    # Define colors for each severity
    colors = {
        '🔴 Critical': '#e74c3c',
        '🟠 High': '#e67e22',
        '🟡 Medium': '#f1c40f',
        '🟢 Low': '#2ecc71'
    }
    
    fig = px.pie(
        values=severity_counts.values,
        names=severity_counts.index,
        title="Issues by Severity",
        color=severity_counts.index,
        color_discrete_map=colors
    )
    return fig


def create_column_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a distribution chart for a specific column."""
    if column not in df.columns:
        return None
    
    if df[column].dtype in ['int64', 'float64']:
        fig = px.histogram(
            df,
            x=column,
            title=f"Distribution of {column}",
            marginal="box"
        )
    else:
        value_counts = df[column].value_counts().head(20)
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Top 20 Values in {column}",
            labels={'x': column, 'y': 'Count'}
        )
    
    return fig


# ============================================
# MAIN APPLICATION UI
# ============================================
st.markdown('<h1 class="main-header">🔍 DataPulse AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated Data Quality Monitoring with AI-Powered Insights</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About DataPulse AI")
    
    st.markdown("""
                **DataPulse AI** automatically detects data quality issues and explains them in plain English using AI.
                
                **Features:**
                - 🔍 Missing value detection
                - 📊 Outlier identification (Z-score)
                - 🔄 Duplicate detection
                - 🏷️ Data type validation
                - 📝 Format consistency checks
                - 🤖 AI-powered explanations
                - 💡 Actionable recommendations
                
                **Powered by:**
                - Google Gemini AI (Free)
                - Python + Streamlit
                - Statistical algorithms
                """)
                
    st.markdown("---")
                
    st.markdown("**🎯 How to Use:**")
    st.markdown("""
                1. Upload your CSV or Excel file
                2. Click 'Analyze Data Quality'
                3. Review AI-generated insights
                4. Export report if needed
                """)
    
    st.markdown("---")
    
    # Generate sample data button
    st.markdown("**📊 Need Sample Data?**")
    if st.button("Generate Sample CSV", use_container_width=True):
        # Create sample data with intentional quality issues
        np.random.seed(42)
        n_rows = 100
        
        sample_df = pd.DataFrame({
            'product_id': range(1, n_rows + 1),
            'product_name': [f'Product_{i}' for i in range(1, n_rows + 1)],
            'category': np.random.choice(['Electronics', 'Clothing', 'electronics', 'CLOTHING', 'Home'], n_rows),
            'price': np.random.normal(100, 30, n_rows),
            'quantity_sold': np.random.randint(1, 100, n_rows),
            'revenue': np.random.normal(5000, 1500, n_rows),
            'date': pd.date_range('2024-01-01', periods=n_rows).astype(str)
        })
        
        # Introduce quality issues
        sample_df.loc[5:15, 'price'] = np.nan  # Missing values
        sample_df.loc[20, 'price'] = 10000  # Outlier
        sample_df.loc[50, 'revenue'] = -5000  # Negative outlier
        sample_df.loc[75, 'quantity_sold'] = 9999  # Outlier
        sample_df.iloc[25:28] = sample_df.iloc[22:25].values  # Duplicates
        
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv,
            file_name="sample_data_with_issues.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    st.caption("Built with ❤️ using Google Gemini + Streamlit")

# Main content area
uploaded_file = st.file_uploader(
    "📁 Upload your dataset (CSV or Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload a CSV or Excel file to analyze data quality"
)

if uploaded_file is not None:
    # Load data
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Successfully loaded: **{len(df):,}** rows × **{len(df.columns)}** columns")
        
        # Show data preview
        with st.expander("👁️ Preview Data (First 10 Rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Show basic statistics
        with st.expander("📊 Basic Statistics", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.metric("Numeric Columns", numeric_cols)
            with col4:
                text_cols = len(df.select_dtypes(include=['object']).columns)
                st.metric("Text Columns", text_cols)
            
            st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔍 Analyze Data Quality",
                type="primary",
                use_container_width=True
            )
        
        if analyze_button:
            with st.spinner("🤖 AI is analyzing your data... This may take 30-60 seconds."):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                all_issues = []
                
                # Run all detections
                status_text.text("Checking for missing values...")
                progress_bar.progress(10)
                missing_issues = detect_missing_values(df)
                all_issues.extend(missing_issues)
                
                status_text.text("Detecting outliers...")
                progress_bar.progress(30)
                outlier_issues = detect_outliers(df)
                all_issues.extend(outlier_issues)
                
                status_text.text("Finding duplicates...")
                progress_bar.progress(50)
                duplicate_issues = detect_duplicates(df)
                all_issues.extend(duplicate_issues)
                
                status_text.text("Checking data types...")
                progress_bar.progress(70)
                type_issues = detect_data_type_issues(df)
                all_issues.extend(type_issues)
                
                status_text.text("Analyzing format consistency...")
                progress_bar.progress(90)
                format_issues = detect_inconsistent_formats(df)
                all_issues.extend(format_issues)
                
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.markdown("---")
            st.header("📋 Data Quality Report")
            
            if not all_issues:
                st.success("🎉 Excellent! No major data quality issues detected.")
                st.balloons()
            else:
                # Summary metrics
                st.subheader("📊 Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                critical_count = len([i for i in all_issues if '🔴' in i['severity']])
                high_count = len([i for i in all_issues if '🟠' in i['severity']])
                medium_count = len([i for i in all_issues if '🟡' in i['severity']])
                low_count = len([i for i in all_issues if '🟢' in i['severity']])
                
                with col1:
                    st.metric("🔴 Critical", critical_count)
                with col2:
                    st.metric("🟠 High", high_count)
                with col3:
                    st.metric("🟡 Medium", medium_count)
                with col4:
                    st.metric("🟢 Low", low_count)
                
                # Visualizations
                st.markdown("---")
                st.subheader("📈 Visual Analysis")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    type_chart = create_issue_summary_chart(all_issues)
                    if type_chart:
                        st.plotly_chart(type_chart, use_container_width=True)
                
                with viz_col2:
                    severity_chart = create_severity_chart(all_issues)
                    if severity_chart:
                        st.plotly_chart(severity_chart, use_container_width=True)
                
                st.markdown("---")
                
                # Display each issue
                st.subheader("🔎 Detailed Findings")
                
                for idx, issue in enumerate(all_issues, 1):
                    with st.expander(
                        f"{issue['severity']} Issue #{idx}: {issue['type']} in '{issue['column']}'",
                        expanded=(idx <= 3)  # Expand first 3 issues
                    ):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**📌 Details:** {issue['description']}")
                            if 'details' in issue:
                                st.markdown(f"**📈 Statistics:** {issue['details']}")
                        
                        with col2:
                            st.markdown(f"**⚠️ Severity:** {issue['severity']}")
                            st.markdown(f"**🔢 Affected:** {issue['count']:,} items")
                        
                        st.markdown("---")
                        st.markdown("**🤖 AI Analysis & Recommendations:**")
                        st.info(issue['ai_explanation'])
                        
                        # Show distribution for the affected column if numeric
                        if issue['column'] != 'All columns' and issue['column'] in df.columns:
                            if df[issue['column']].dtype in ['int64', 'float64']:
                                with st.expander("📊 View Column Distribution"):
                                    dist_fig = create_column_distribution(df, issue['column'])
                                    if dist_fig:
                                        st.plotly_chart(dist_fig, use_container_width=True)
                
                # Export report
                st.markdown("---")
                st.subheader("📥 Export Report")
                
                # Generate text report
                report_text = f"""DATA QUALITY REPORT
                            {'='*60}
                            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                            Tool: DataPulse AI (Powered by Google Gemini)
                            {'='*60}

                            DATASET INFORMATION:
                            - File: {uploaded_file.name}
                            - Total Rows: {len(df):,}
                            - Total Columns: {len(df.columns)}
                            - Columns: {', '.join(df.columns)}

                            SUMMARY:
                            - Critical Issues: {critical_count}
                            - High Priority: {high_count}
                            - Medium Priority: {medium_count}
                            - Low Priority: {low_count}
                            - Total Issues: {len(all_issues)}

                            {'='*60}
                            DETAILED FINDINGS
                            {'='*60}
                            """
                
                for idx, issue in enumerate(all_issues, 1):
                    report_text += f"""
{'='*60}
ISSUE #{idx}: {issue['type']} - {issue['column']}
{'='*60}

Severity: {issue['severity']}
Description: {issue['description']}
Affected Items: {issue['count']:,}

AI ANALYSIS:
{issue['ai_explanation']}

"""
                
                report_text += f"""
{'='*60}
END OF REPORT
{'='*60}
"""
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📄 Download Text Report",
                        data=report_text,
                        file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col2:
                    # Create CSV summary
                    issues_df = pd.DataFrame([{
                        'Issue #': idx,
                        'Type': i['type'],
                        'Column': i['column'],
                        'Severity': i['severity'],
                        'Count': i['count'],
                        'Percentage': f"{i['percentage']:.1f}%" if i['percentage'] > 0 else 'N/A',
                        'Description': i['description']
                    } for idx, i in enumerate(all_issues, 1)])
                    
                    csv_export = issues_df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV Summary",
                        data=csv_export,
                        file_name=f"issues_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        st.info("Please make sure your file is a valid CSV or Excel file with proper encoding.")

else:
    # Welcome message when no file is uploaded
    st.info("👆 Upload a dataset to get started!")
    
    st.markdown("""
    ### What DataPulse AI Will Check:
    
    | Check | Description |
    |-------|-------------|
    | **Missing Values** | Identifies gaps in your data with business impact analysis |
    | **Outliers** | Detects unusual values using statistical methods (Z-score) |
    | **Duplicates** | Finds duplicate records that could skew your analysis |
    | **Data Types** | Verifies columns are stored in the correct format |
    | **Format Consistency** | Identifies inconsistent formatting (e.g., case variations) |
    
    ### For Each Issue, You Get:
    - ✅ Severity assessment (Critical/High/Medium/Low)
    - ✅ Business impact explanation in plain English
    - ✅ Likely root cause analysis
    - ✅ Specific actionable recommendations
    - ✅ Visual distributions where applicable
    
    ### Try it now with sample data or upload your own!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Google Gemini (Free AI) + Streamlit + Python</p>
    <p>🌟 No API costs | 🚀 100% Free Tools | 💯 Production Ready</p>
    <p><small>Using NEW google-genai SDK (Updated January 2025)</small></p>
</div>
""", unsafe_allow_html=True)