import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

# =========================================
# LOAD SAVED FILES
# =========================================

model = pickle.load(
    open('models/decision_tree_model.pkl', 'rb')
)

columns = pickle.load(
    open('models/columns.pkl', 'rb')
)

# =========================================
# TITLE
# =========================================

st.title("🏥 Medical Insurance Cost Prediction")

st.markdown("""
Predict medical insurance charges using a trained Decision Tree Regressor.
""")

st.write("---")



# =========================================
# USER INPUT SECTION
# =========================================

st.header("📝 Enter Customer Details")

col1, col2 = st.columns(2)

# =========================================
# COLUMN 1
# =========================================

with col1:

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Sex",
        ["male", "female"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0
    )

    children = st.slider(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=1
    )

# =========================================
# COLUMN 2
# =========================================

with col2:

    smoker = st.selectbox(
        "Smoker",
        ["yes", "no"]
    )

    region = st.selectbox(
        "Region",
        [
            "southwest",
            "southeast",
            "northwest",
            "northeast"
        ]
    )

# =========================================
# FEATURE ENGINEERING
# =========================================

# BMI Category

if bmi < 18.5:
    bmi_category = "Underweight"

elif bmi < 25:
    bmi_category = "Normal"

elif bmi < 30:
    bmi_category = "Overweight"

else:
    bmi_category = "Obese"

# Health Risk

health_risk = age * bmi

# High Risk Smoker

high_risk_smoker = 1 if (
    smoker == "yes" and bmi > 30
) else 0

# =========================================
# CREATE INPUT DATAFRAME
# =========================================

input_data = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'health_risk': health_risk,
    'high_risk_smoker': high_risk_smoker
}

input_df = pd.DataFrame([input_data])

# =========================================
# HANDLE DUMMY VARIABLES
# =========================================

# Sex
sex_col = f"sex_{sex}"

if sex_col in columns:
    input_df[sex_col] = 1

# Smoker
smoker_col = f"smoker_{smoker}"

if smoker_col in columns:
    input_df[smoker_col] = 1

# Region
region_col = f"region_{region}"

if region_col in columns:
    input_df[region_col] = 1

# BMI Category
bmi_col = f"bmi_category_{bmi_category}"

if bmi_col in columns:
    input_df[bmi_col] = 1

# =========================================
# ADD MISSING COLUMNS
# =========================================

for col in columns:

    if col not in input_df.columns:
        input_df[col] = 0

# =========================================
# COLUMN ORDER
# =========================================

input_df = input_df[columns]

# =========================================
# PREDICTION
# =========================================

if st.button("🚀 Predict Insurance Cost"):

    prediction = model.predict(input_df)

    predicted_cost = prediction[0]

    st.success(
        f"Estimated Insurance Charges: ₹ {predicted_cost:,.2f}"
    )

    # Risk Interpretation

    if predicted_cost < 10000:

        st.info("🟢 Low Insurance Risk")

    elif predicted_cost < 30000:

        st.warning("🟡 Medium Insurance Risk")

    else:

        st.error("🔴 High Insurance Risk")

    st.balloons()

# =========================================
# SHOW INPUT DATA
# =========================================

with st.expander("📂 View Processed Input Data"):

    st.dataframe(input_df)

# =========================================
# FOOTER
# =========================================

st.write("---")

st.markdown("""
### 👩‍💻 Project Details

- Problem Type: Regression
- Model: Decision Tree Regressor
- Domain: Healthcare Analytics
- Deployment: Streamlit

Built as an end-to-end Machine Learning project.
""")