import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration with sleek modern theme
st.set_page_config(
    page_title="UDISE+ School Quality Dashboard",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #1e293b;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 5px;
    }
    .status-badge {
        font-size: 1.5rem;
        font-weight: 700;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
        color: white;
    }
</style>
""", unsafe_allowed_html=True)

# Helper function to load model pipeline
@st.cache_resource
def load_pipeline():
    pipeline_path = 'school_classifier_pipeline.joblib'
    if not os.path.exists(pipeline_path):
        return None
    return joblib.load(pipeline_path)

pipeline = load_pipeline()

st.markdown("<h1 class='main-title'>🏫 UDISE+ School Quality Diagnostics</h1>", unsafe_allowed_html=True)
st.markdown("<p class='subtitle'>Predictive analytics & policy alignment checks for government schools</p>", unsafe_allowed_html=True)

if pipeline is None:
    st.error("⚠️ **Model pipeline file not found!**")
    st.info("""
    To generate the model file:
    1. Make sure `final_merged_data.csv` is present in the workspace directory.
    2. Run the main training script: `python udise_school_classification.py`.
    3. Once completed, `school_classifier_pipeline.joblib` will be generated, and this app will load it automatically.
    """)
    st.stop()

# Load pipeline objects
scaler = pipeline['scaler']
pca = pipeline['pca']
label_encoders = pipeline['label_encoders']
feature_cols = pipeline['feature_cols']
train_medians = pipeline['train_medians']
voting_model = pipeline['voting_model']

# Tabs layout
tab1, tab2, tab3 = st.tabs(["📊 Quality Diagnosis Tool", "📄 Policy Benchmarks & FAQ", "📈 Batch Predict"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📝 Enter School Characteristics")
        st.write("Modify the indicators below to diagnose the school:")
        
        user_inputs = {}
        
        # Dynamically render inputs based on trained feature columns
        for col in feature_cols:
            if col in label_encoders:
                # Categorical Feature Input
                le = label_encoders[col]
                user_inputs[col] = st.selectbox(
                    f"{col.replace('_', ' ').title()}",
                    options=list(le.classes_),
                    key=f"input_{col}"
                )
            else:
                # Numeric Feature Input
                col_title = col.replace('_', ' ').title()
                
                # Custom ranges and defaults for numeric features
                if 'ratio' in col or 'score' in col:
                    max_val = 100.0
                    val = 50.0
                    step = 0.5
                elif 'count' in col or 'student' in col:
                    max_val = 5000.0
                    val = 250.0
                    step = 5.0
                elif 'teacher' in col or 'regular' in col or 'graduate' in col:
                    max_val = 100.0
                    val = 5.0
                    step = 1.0
                elif 'classroom' in col:
                    max_val = 100.0
                    val = 8.0
                    step = 1.0
                elif 'available' in col or 'functional' in col or 'panel' in col or 'harvest' in col or 'avail' in col:
                    # Categorical codes like 1/2 or 0/1/2 representing facilities
                    max_val = 3.0
                    val = 1.0
                    step = 1.0
                else:
                    max_val = 500.0
                    val = 0.0
                    step = 1.0
                    
                user_inputs[col] = st.number_input(
                    col_title,
                    min_value=0.0,
                    max_value=max_val,
                    value=float(val),
                    step=float(step),
                    key=f"input_{col}"
                )
                
    with col2:
        st.markdown("### 🔍 Model Diagnostic Results")
        
        # Calculate diagnostics behind the scenes to show inspectors
        # Note: if raw counts are entered, we approximate the scores for visual diagnostic guidance
        total_teacher = user_inputs.get('total_teacher', 5.0)
        total_student = user_inputs.get('total_student_count', 150.0)
        
        # Student-Teacher Ratio Score
        if total_teacher == 0:
            calc_ratio_score = 0.0
            calc_quality_score = 0.0
            calc_effectiveness_score = 0.0
        else:
            ratio = total_student / total_teacher
            ideal = 40.0 # primary benchmark
            deviation = (abs(ratio - ideal) / ideal) / 2
            calc_ratio_score = max(0.0, (1.0 - deviation) * 100.0)
            
            # Approximate quality score using regular teacher counts if available
            reg = user_inputs.get('regular', total_teacher)
            calc_quality_score = min((reg / total_teacher) * 100.0, 100.0)
            calc_effectiveness_score = (0.7 * calc_ratio_score) + (0.3 * calc_quality_score)
            
        # Infrastructure score approximation (based on facility selectboxes)
        calc_infra_score = 0.0
        infra_keys = ['classrooms_in_good_condition', 'drinking_water_functional', 'library_availability', 'playground_available']
        found_keys = [k for k in infra_keys if k in user_inputs]
        if found_keys:
            # Simple average representing scale
            points = 0
            for k in found_keys:
                v = user_inputs[k]
                if v == 1.0 or v == 1: # Yes / Good
                    points += 25
                elif v == 2.0 or v == 2: # No / Repair
                    points += 5
                else:
                    points += 15
            calc_infra_score = min(points * (100 / (len(found_keys) * 25)), 100.0)
        else:
            calc_infra_score = 35.0 # default diagnostic placeholder
            
        # Rule-based diagnostic label
        rule_odd = False
        odd_reasons = []
        if calc_effectiveness_score < 30:
            rule_odd = True
            odd_reasons.append(f"Teacher Effectiveness Score is below benchmark ({calc_effectiveness_score:.1f} < 30)")
        if calc_infra_score < 15:
            rule_odd = True
            odd_reasons.append(f"Infrastructure Score is below benchmark ({calc_infra_score:.1f} < 15)")

        # Prepare input row for model prediction
        input_df = pd.DataFrame([user_inputs])
        
        # Categorical encode using label encoders
        for col in feature_cols:
            if col in label_encoders:
                le = label_encoders[col]
                # Map value to encoder class, fallback to -1 if unseen
                val = input_df[col].iloc[0]
                if val in le.classes_:
                    input_df[col] = le.transform([val])[0]
                else:
                    input_df[col] = -1
                    
        # Impute missing values
        input_df = input_df.fillna(train_medians)
        
        # Scaling
        input_scaled = scaler.transform(input_df)
        
        # PCA Projection
        input_pca = pca.transform(input_scaled)
        
        # Prediction
        prediction = voting_model.predict(input_pca)[0]
        prob_odd = voting_model.predict_proba(input_pca)[0][1]
        
        # Display decision results
        if prediction == 0:
            st.markdown("<div class='status-badge' style='background-color: #10b981;'>✅ STANDARDIZED SCHOOL QUALITY</div>", unsafe_allowed_html=True)
            st.success("The machine learning model predicts this school meets the general government resource standard.")
        else:
            st.markdown("<div class='status-badge' style='background-color: #f97316;'>⚠️ ODD SCHOOL QUALITY (Vulnerable)</div>", unsafe_allowed_html=True)
            st.warning("The machine learning model classifies this school as **Odd** (struggling resource/capacity). Recommended for immediate intervention.")

        # Metric cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{prob_odd*100:.1f}%</div>
                <div class='metric-label'>Odd Probability</div>
            </div>
            """, unsafe_allowed_html=True)
        with m_col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{calc_effectiveness_score:.1f}</div>
                <div class='metric-label'>Teacher Score (Estimated)</div>
            </div>
            """, unsafe_allowed_html=True)
        with m_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{calc_infra_score:.1f}</div>
                <div class='metric-label'>Infra Score (Estimated)</div>
            </div>
            """, unsafe_allowed_html=True)
            
        # Policy Alignment Diagnosis
        st.markdown("---")
        st.markdown("### 📋 Policy Check Summary")
        if rule_odd:
            st.markdown("❌ **Rule-based benchmarks fail due to:**")
            for r in odd_reasons:
                st.markdown(f"- {r}")
        else:
            st.markdown(r"✔ **All rule-based thresholds are met** (Teacher Score $\geq 30$ and Infrastructure Score $\geq 15$).")

with tab2:
    st.markdown("### 📄 Project Assumptions & Samagra Shiksha Benchmarks")
    st.write("This machine learning pipeline classifies schools into **Standardized** or **Odd** categories using realistic capacity thresholds:")
    
    st.markdown(r"""
    *   **Task Type**: Binary Classification.
    *   **Class Definition**:
        *   `Standardized (0)`: Meets basic policy guidelines.
        *   `Odd (1)`: Outlier class indicating severe resource constraints.
    *   **Government-Tuned Thresholds (MDS21 Sprint Sync)**:
        *   **Teacher Effectiveness Score** < 30 $\rightarrow$ Odd. Focuses 70% on Student-Teacher Ratio (ideal primary ratio 40:1) and 30% on qualifications.
        *   **Infrastructure Score** < 15 $\rightarrow$ Odd. Evaluates 12 core functional facilities (water, handwash, electricity, classrooms repair, etc.).
    *   **Normality & Test Results**: Shapiro-Wilk and Q-Q Plot verification showed non-normality in scores, necessitating the non-parametric Mann-Whitney U test which proved infrastructure significantly differs between classes ($p < 0.05$).
    """)

with tab3:
    st.markdown("### 📂 Upload Batch Dataset")
    st.write("Upload a CSV file of schools containing raw indicators to run batch predictions:")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Loaded batch dataset:", batch_df.head())
        
        # Check if required columns are present
        missing_cols = [c for c in feature_cols if c not in batch_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            if st.button("Run Batch Prediction"):
                # Preprocess batch
                proc_df = batch_df[feature_cols].copy()
                
                # Encoding
                for col in feature_cols:
                    if col in label_encoders:
                        le = label_encoders[col]
                        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                        proc_df[col] = proc_df[col].astype(str).apply(lambda x: le_dict.get(x, -1))
                
                # Imputation & Scaling
                proc_df = proc_df.fillna(train_medians)
                proc_scaled = scaler.transform(proc_df)
                
                # PCA Projection & Prediction
                proc_pca = pca.transform(proc_scaled)
                preds = voting_model.predict(proc_pca)
                probs = voting_model.predict_proba(proc_pca)[:, 1]
                
                # Add columns to original dataframe
                result_df = batch_df.copy()
                result_df['Predicted_Label'] = preds
                result_df['Predicted_Odd_Probability'] = probs
                result_df['Status'] = result_df['Predicted_Label'].map({0: 'Standardized', 1: 'Odd'})
                
                st.markdown("### 🎯 Prediction Results")
                st.write(result_df[['state', 'district', 'total_student_count', 'total_teacher', 'Status', 'Predicted_Odd_Probability']].head(10))
                
                # Download CSV button
                csv_data = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv_data,
                    file_name="school_predictions_output.csv",
                    mime="text/csv"
                )
