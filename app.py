import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import io
import base64
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="Professional Data Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

# Main title
st.markdown('<h1 class="main-header">📊 Professional Data Analytics Suite</h1>', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose Analysis Mode",
    ["Data Upload", "Data Overview", "Descriptive Statistics", "Inferential Statistics", "Regression Analysis", "Data Visualization"]
)

# Theme selector
theme = st.sidebar.selectbox("Select Theme", ["Light", "Dark"])

# Data upload section
if app_mode == "Data Upload":
    st.header("📤 Data Upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Supported formats: CSV, Excel (.xlsx, .xls)"
        )
        
        if uploaded_file is not None:
            try:
                # Read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Display basic info
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10))
                
                # Data info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", df.shape[0])
                with col2:
                    st.metric("Columns", df.shape[1])
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with col2:
        st.subheader("Data Requirements")
        st.info("""
        **Supported Formats:**
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        
        **Recommendations:**
        - First row should contain column headers
        - Use consistent data types per column
        - Handle missing values appropriately
        """)

# Data overview section
elif app_mode == "Data Overview":
    if st.session_state.data is None:
        st.warning("Please upload a dataset first!")
        st.stop()
    
    df = st.session_state.data
    
    st.header("📋 Data Overview")
    
    # Dataset information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        info_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': df.nunique()
        })
        st.dataframe(info_df)
    
    with col2:
        st.subheader("Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap='viridis')
        st.pyplot(fig)
    
    # Data quality metrics
    st.subheader("Data Quality Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">Completeness<br><b>{:.1f}%</b></div>'.format(
            (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        ), unsafe_allow_html=True)
    
    with col2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.markdown('<div class="metric-card">Numeric Columns<br><b>{}</b></div>'.format(
            len(numeric_cols)
        ), unsafe_allow_html=True)
    
    with col3:
        categorical_cols = df.select_dtypes(include=['object']).columns
        st.markdown('<div class="metric-card">Categorical Columns<br><b>{}</b></div>'.format(
            len(categorical_cols)
        ), unsafe_allow_html=True)
    
    with col4:
        duplicate_rows = df.duplicated().sum()
        st.markdown('<div class="metric-card">Duplicate Rows<br><b>{}</b></div>'.format(
            duplicate_rows
        ), unsafe_allow_html=True)
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.sample(min(20, len(df))))

# Descriptive statistics section
elif app_mode == "Descriptive Statistics":
    if st.session_state.data is None:
        st.warning("Please upload a dataset first!")
        st.stop()
    
    df = st.session_state.data
    
    st.header("📈 Descriptive Statistics")
    
    # Column selection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_cols = st.multiselect(
        "Select columns for analysis:",
        numeric_cols,
        default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
    )
    
    if selected_cols:
        # Basic statistics
        st.subheader("Basic Statistics")
        desc_stats = df[selected_cols].describe()
        st.dataframe(desc_stats)
        
        # Additional statistics
        st.subheader("Additional Statistics")
        additional_stats = pd.DataFrame({
            'Column': selected_cols,
            'Skewness': [df[col].skew() for col in selected_cols],
            'Kurtosis': [df[col].kurtosis() for col in selected_cols],
            'Variance': [df[col].var() for col in selected_cols],
            'Range': [df[col].max() - df[col].min() for col in selected_cols]
        })
        st.dataframe(additional_stats)
        
        # Correlation matrix
        if len(selected_cols) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[selected_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        st.subheader("Distribution Analysis")
        for col in selected_cols[:4]:  # Limit to first 4 columns
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, y=col, title=f"Box Plot of {col}")
                st.plotly_chart(fig, use_container_width=True)

# Inferential statistics section
elif app_mode == "Inferential Statistics":
    if st.session_state.data is None:
        st.warning("Please upload a dataset first!")
        st.stop()
    
    df = st.session_state.data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.header("🔬 Inferential Statistics")
    
    # Test selection
    test_type = st.selectbox(
        "Select Statistical Test:",
        ["One-Sample T-Test", "Two-Sample T-Test", "Paired T-Test", "Chi-Square Test", "ANOVA"]
    )
    
    if test_type == "One-Sample T-Test":
        st.subheader("One-Sample T-Test")
        
        col1, col2 = st.columns(2)
        with col1:
            test_column = st.selectbox("Select column:", numeric_cols)
            test_value = st.number_input("Test value:", value=0.0)
        
        with col2:
            alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
        
        if st.button("Run One-Sample T-Test"):
            data_clean = df[test_column].dropna()
            t_stat, p_value = stats.ttest_1samp(data_clean, test_value)
            
            result_html = f"""
            <div class="{'success-box' if p_value < alpha else 'warning-box'}">
                <h4>One-Sample T-Test Results</h4>
                <p><strong>T-statistic:</strong> {t_stat:.4f}</p>
                <p><strong>P-value:</strong> {p_value:.4f}</p>
                <p><strong>Significance level:</strong> {alpha}</p>
                <p><strong>Result:</strong> {'Reject null hypothesis' if p_value < alpha else 'Fail to reject null hypothesis'}</p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
    
    elif test_type == "Two-Sample T-Test":
        st.subheader("Two-Sample T-Test")
        
        col1, col2 = st.columns(2)
        with col1:
            col1_select = st.selectbox("Select first column:", numeric_cols)
            col2_select = st.selectbox("Select second column:", numeric_cols)
        
        with col2:
            alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
            equal_var = st.checkbox("Assume equal variances", value=True)
        
        if st.button("Run Two-Sample T-Test"):
            data1 = df[col1_select].dropna()
            data2 = df[col2_select].dropna()
            
            t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var)
            
            result_html = f"""
            <div class="{'success-box' if p_value < alpha else 'warning-box'}">
                <h4>Two-Sample T-Test Results</h4>
                <p><strong>T-statistic:</strong> {t_stat:.4f}</p>
                <p><strong>P-value:</strong> {p_value:.4f}</p>
                <p><strong>Significance level:</strong> {alpha}</p>
                <p><strong>Equal variances:</strong> {equal_var}</p>
                <p><strong>Result:</strong> {'Reject null hypothesis' if p_value < alpha else 'Fail to reject null hypothesis'}</p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
    
    elif test_type == "ANOVA":
        st.subheader("One-Way ANOVA")
        
        selected_cols = st.multiselect(
            "Select columns for ANOVA:",
            numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        
        alpha = st.slider("Significance level (α):", 0.01, 0.10, 0.05)
        
        if len(selected_cols) >= 2 and st.button("Run ANOVA"):
            groups = [df[col].dropna() for col in selected_cols]
            f_stat, p_value = stats.f_oneway(*groups)
            
            result_html = f"""
            <div class="{'success-box' if p_value < alpha else 'warning-box'}">
                <h4>One-Way ANOVA Results</h4>
                <p><strong>F-statistic:</strong> {f_stat:.4f}</p>
                <p><strong>P-value:</strong> {p_value:.4f}</p>
                <p><strong>Significance level:</strong> {alpha}</p>
                <p><strong>Groups:</strong> {', '.join(selected_cols)}</p>
                <p><strong>Result:</strong> {'Reject null hypothesis' if p_value < alpha else 'Fail to reject null hypothesis'}</p>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)

# Regression analysis section
elif app_mode == "Regression Analysis":
    if st.session_state.data is None:
        st.warning("Please upload a dataset first!")
        st.stop()
    
    df = st.session_state.data
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    st.header("📊 Regression Analysis")
    
    # Variable selection
    col1, col2 = st.columns(2)
    
    with col1:
        dependent_var = st.selectbox("Select dependent variable (Y):", numeric_cols)
    
    with col2:
        independent_vars = st.multiselect(
            "Select independent variables (X):",
            [col for col in numeric_cols if col != dependent_var],
            default=[col for col in numeric_cols if col != dependent_var][:3]
        )
    
    if dependent_var and independent_vars:
        if st.button("Run Regression Analysis"):
            # Prepare data
            y = df[dependent_var].dropna()
            X = df[independent_vars].dropna()
            
            # Align indices
            common_idx = y.index.intersection(X.index)
            y = y.loc[common_idx]
            X = X.loc[common_idx]
            
            # Add constant
            X = sm.add_constant(X)
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Summary")
                st.text(str(model.summary()))
            
            with col2:
                st.subheader("Key Metrics")
                
                metrics_html = f"""
                <div class="analysis-section">
                    <h4>Model Performance</h4>
                    <p><strong>R-squared:</strong> {model.rsquared:.4f}</p>
                    <p><strong>Adjusted R-squared:</strong> {model.rsquared_adj:.4f}</p>
                    <p><strong>F-statistic:</strong> {model.fvalue:.4f}</p>
                    <p><strong>F-statistic p-value:</strong> {model.f_pvalue:.4f}</p>
                    <p><strong>AIC:</strong> {model.aic:.4f}</p>
                    <p><strong>BIC:</strong> {model.bic:.4f}</p>
                </div>
                """
                st.markdown(metrics_html, unsafe_allow_html=True)
                
                # Coefficients
                st.subheader("Coefficients")
                coef_df = pd.DataFrame({
                    'Variable': model.params.index,
                    'Coefficient': model.params.values,
                    'P-value': model.pvalues.values,
                    'Significant': model.pvalues.values < 0.05
                })
                st.dataframe(coef_df)
            
            # Diagnostic plots
            st.subheader("Diagnostic Plots")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residuals vs Fitted
                fig = px.scatter(
                    x=model.fittedvalues, 
                    y=model.resid,
                    title="Residuals vs Fitted Values"
                )
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Q-Q plot
                fig = go.Figure()
                stats.probplot(model.resid, dist="norm", plot=None)
                theoretical_q, sample_q = stats.probplot(model.resid, dist="norm", plot=None)
                fig.add_trace(go.Scatter(x=theoretical_q, y=sample_q, mode='markers', name='Data'))
                fig.add_trace(go.Scatter(x=theoretical_q, y=theoretical_q, mode='lines', name='Normal'))
                fig.update_layout(title="Q-Q Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
                st.plotly_chart(fig, use_container_width=True)

# Data visualization section
elif app_mode == "Data Visualization":
    if st.session_state.data is None:
        st.warning("Please upload a dataset first!")
        st.stop()
    
    df = st.session_state.data
    
    st.header("📊 Data Visualization")
    
    # Chart type selection
    chart_type = st.selectbox(
        "Select visualization type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Heatmap", "Pair Plot"]
    )
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if chart_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis:", numeric_cols)
            y_axis = st.selectbox("Y-axis:", numeric_cols)
        with col2:
            color_col = st.selectbox("Color by:", [None] + categorical_cols)
            size_col = st.selectbox("Size by:", [None] + numeric_cols)
        
        fig = px.scatter(
            df, x=x_axis, y=y_axis, 
            color=color_col, size=size_col,
            title=f"{y_axis} vs {x_axis}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Line Chart":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis:", df.columns)
            y_axis = st.selectbox("Y-axis:", numeric_cols)
        with col2:
            color_col = st.selectbox("Color by:", [None] + categorical_cols)
        
        fig = px.line(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} over {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Bar Chart":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("X-axis:", categorical_cols + numeric_cols)
            y_axis = st.selectbox("Y-axis:", numeric_cols)
        with col2:
            color_col = st.selectbox("Color by:", [None] + categorical_cols)
        
        fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Column:", numeric_cols)
            bins = st.slider("Number of bins:", 10, 100, 30)
        with col2:
            color_col = st.selectbox("Color by:", [None] + categorical_cols)
        
        fig = px.histogram(df, x=x_axis, nbins=bins, color=color_col, title=f"Distribution of {x_axis}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Box Plot":
        col1, col2 = st.columns(2)
        with col1:
            y_axis = st.selectbox("Y-axis:", numeric_cols)
            x_axis = st.selectbox("X-axis (optional):", [None] + categorical_cols)
        with col2:
            color_col = st.selectbox("Color by:", [None] + categorical_cols)
        
        fig = px.box(df, x=x_axis, y=y_axis, color=color_col, title=f"Box Plot of {y_axis}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Heatmap":
        selected_cols = st.multiselect(
            "Select columns for correlation heatmap:",
            numeric_cols,
            default=numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
        )
        
        if selected_cols:
            corr_matrix = df[selected_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Pair Plot":
        selected_cols = st.multiselect(
            "Select columns for pair plot:",
            numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
        )
        
        if selected_cols and len(selected_cols) >= 2:
            fig = px.scatter_matrix(df, dimensions=selected_cols, title="Pair Plot")
            st.plotly_chart(fig, use_container_width=True)

