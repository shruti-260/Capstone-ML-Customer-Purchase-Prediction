import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="Purchase Predictor", page_icon="🛒", layout="wide")

st.markdown("""
<h1 style='text-align: center; color: #4CAF50;'>🛒 Customer Purchase Prediction</h1>
<p style='text-align: center;'>AI model predicting whether a customer will buy</p>
<hr>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
rf_model = joblib.load("rf_model.pkl")
model_features = rf_model.feature_names_in_

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📄 Page Activity")
    administrative = st.number_input("Administrative Pages", 0)
    informational = st.number_input("Informational Pages", 0)
    product_related = st.number_input("Product Pages", 0)

    st.subheader("⏱ Duration (seconds)")
    admin_dur = st.number_input("Administrative Duration", 0.0)
    info_dur = st.number_input("Informational Duration", 0.0)
    prod_dur = st.number_input("Product Duration", 0.0)

with col2:
    st.subheader("📊 Behavior Metrics")
    bounce = st.slider("Bounce Rate", 0.0, 1.0, 0.1)
    exit_rate = st.slider("Exit Rate", 0.0, 1.0, 0.1)
    special_day = st.slider("Special Day Value", 0.0, 1.0, 0.0)

    st.subheader("👤 Visitor Info")
    visitor = st.selectbox("Visitor Type", ["New_Visitor", "Returning_Visitor", "Other"])
    weekend = st.selectbox("Weekend", [0, 1])
    month = st.selectbox("Month", ['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'])

# ---------------- FEATURE ENGINEERING ----------------
total_pages = administrative + informational + product_related
total_time = admin_dur + info_dur + prod_dur

avg_time = total_time / (total_pages + 1)
prod_ratio = product_related / (total_pages + 1)
is_returning = int(visitor == "Returning_Visitor")
month_code = ['Jan','Feb','Mar','Apr','May','June','Jul','Aug','Sep','Oct','Nov','Dec'].index(month)

# ---------------- SESSION SUMMARY ----------------
st.markdown("### 📈 Session Summary")
m1, m2, m3 = st.columns(3)
m1.metric("Total Pages", total_pages)
m2.metric("Total Time Spent", f"{total_time:.1f}s")
m3.metric("Avg Time/Page", f"{avg_time:.2f}")

# ---------------- SESSION CHARTS ----------------
st.markdown("### 📊 Session Activity Distribution")

activity_df = pd.DataFrame({
    "Page Type": ["Administrative", "Informational", "Product"],
    "Pages Viewed": [administrative, informational, product_related]
})
st.bar_chart(activity_df.set_index("Page Type"))

st.markdown("### ⏱ Time Spent Distribution")

time_df = pd.DataFrame({
    "Page Type": ["Administrative", "Informational", "Product"],
    "Time Spent": [admin_dur, info_dur, prod_dur]
})
st.bar_chart(time_df.set_index("Page Type"))

# ---------------- CREATE MODEL INPUT ----------------
input_df = pd.DataFrame(0, index=[0], columns=model_features)

def set_feature(name, value):
    if name in model_features:
        input_df[name] = value

set_feature('Avg_Time_Per_Page', avg_time)
set_feature('Product_Page_Ratio', prod_ratio)
set_feature('High_Bounce', int(bounce > 0.2))
set_feature('High_Exit', int(exit_rate > 0.2))
set_feature('Is_Returning_Visitor', is_returning)
set_feature('Weekend_Flag', weekend)
set_feature('Month_Code', month_code)

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Purchase", use_container_width=True):

    prob = rf_model.predict_proba(input_df)[:,1][0]

    st.markdown("## 🎯 Prediction Result")

    if prob >= 0.4:
        st.success(f"### ✅ Purchase Likely\n**Probability:** {prob:.2%}")
    else:
        st.error(f"### ❌ Purchase Not Likely\n**Probability:** {prob:.2%}")

    st.progress(int(prob * 100))

    # -------- Probability Chart --------
    st.markdown("### 📊 Purchase Probability Chart")
    fig, ax = plt.subplots()
    ax.bar(["Not Purchase", "Purchase"], [1-prob, prob])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)
