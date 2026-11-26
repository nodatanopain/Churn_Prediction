import streamlit as st
import pandas as pd
import pickle

# 1. Page Configuration
st.set_page_config(
    page_title="HR Attrition Prediction App",
    page_icon="👨‍💼",
    layout="centered"
)

# 2. Load the trained scikit-learn pipeline model safely
@st.cache_resource
def load_model():
    try:
        with open('hr_deployment_model', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("❌ Model file ('hr_deployment_model.pkl') not found. Please verify the file path.")
        st.stop()

pipeline_model = load_model()

# 3. Streamlit User Interface
st.title("📊 HR Employee Attrition Prediction")
st.write("Enter the employee details below to predict whether they are likely to leave the company.")

# Mapping dictionaries for better UX
dept_mapping = {
    'Sales': 'sales',
    'Technical': 'technical',
    'Support': 'support',
    'IT': 'IT',
    'Product Management': 'product_mng',
    'Marketing': 'marketing',
    'Accounting': 'accounting',
    'Management': 'management',
    'HR': 'hr',
    'R&D': 'RandD'
}

# Create input form for the user
with st.form("prediction_form"):
    st.subheader("Employee Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01)
        last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.6, 0.01)
        number_project = st.number_input("Number of Projects", min_value=1, max_value=10, value=3)
        average_montly_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, value=200)
        time_spend_company = st.number_input("Years at Company (Time Spend)", min_value=1, max_value=10, value=3)
        
    with col2:
        work_accident = st.selectbox("Work Accident History", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        promotion_last_5years = st.selectbox("Promotion in Last 5 Years", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        dept_display = st.selectbox("Department", options=list(dept_mapping.keys()))
        departments = dept_mapping[dept_display]
        
        salary = st.selectbox("Salary Level", options=['low', 'medium', 'high'], format_func=lambda x: x.capitalize())

    # Submit button
    submitted = st.form_submit_button("Predict Attrition", use_container_width=True)

# 4. Handle prediction when the button is clicked
if submitted:
    input_data = pd.DataFrame([{
        'satisfaction_level': satisfaction_level,
        'last_evaluation': last_evaluation,
        'number_project': number_project,
        'average_montly_hours': average_montly_hours,
        'time_spend_company': time_spend_company,
        'work_accident': work_accident,
        'promotion_last_5years': promotion_last_5years,
        'departments': departments,
        'salary': salary
    }])

    prediction = pipeline_model.predict(input_data)
    probability = pipeline_model.predict_proba(input_data)

    st.markdown("---")
    st.subheader("Prediction Results")
    
    attrition_prob = probability[0][1] * 100
    
    if prediction[0] == 1:
        st.error("⚠️ **High Risk:** This employee is likely to **LEAVE** the company.")
    else:
        st.success("✅ **Low Risk:** This employee is likely to **STAY** in the company.")
        
    st.metric(label="Attrition Probability", value=f"{attrition_prob:.2f}%")
    st.progress(int(attrition_prob))