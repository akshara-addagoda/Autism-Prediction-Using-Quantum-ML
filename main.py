import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
from io import BytesIO

import plotly.graph_objects as go
import plotly.express as px

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from classical_ml.svm_model import run_svm
from classical_ml.logistic_model import run_logistic
from classical_ml.xgb_model import run_xgb
from quantum_ml.vqc_model import run_vqc
from quantum_ml.qsvm_model import run_qsvm
from utils.data_loader import load_data

# ============================
# Page Config
# ============================
st.set_page_config(page_title="Autism Risk Screening", layout="centered")

st.title("Early Autism Risk Screening Using Quantum Machine Learning")
st.caption("Behavioural screening for toddlers (12–60 months)")

# ============================
# Sidebar
# ============================
st.sidebar.title("Model Selection")

model_choice = st.sidebar.radio(
    "Select Model",
    ["SVM", "Logistic Regression", "XGBoost", "Quantum VQC", "Quantum SVM (QSVM)"]
)

# ============================
# Gauge
# ============================
def show_risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Autism Risk Score (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 25], 'color': "lightgreen"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"},
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# ============================
# Inputs (UPDATED)
# ============================
st.subheader("Toddler Details")

age = st.number_input("Age (months)", 12, 60, 24)

gender = st.selectbox(
    "Gender",
    ["Male", "Female", "Prefer not to say"]
)

weight = st.number_input("Weight (kg)", 5.0, 30.0, 12.0)

gestation = st.selectbox(
    "Gestational Age at Birth",
    ["Full-term", "Preterm", "Unknown"]
)

family_history = st.radio(
    "Family history of Autism?",
    ["No", "Yes", "Unknown"]
)

# ============================
# Questionnaire
# ============================
st.subheader("Behavioural Questionnaire")

questions = [
    "Eye contact", "Response to name", "Social interaction",
    "Sensitivity", "Understanding emotions", "Routine disturbance",
    "Repetitive behaviour", "Communication delay",
    "Crowd discomfort", "Plays alone"
]

responses = []
for i, q in enumerate(questions):
    responses.append(st.slider(f"Q{i+1}: {q}", 1, 10, 5))

responses = np.array(responses)

# ============================
# Prediction
# ============================
if st.button("Generate Report"):

    with st.spinner("Running models..."):
        start = time.time()

        user_input = (responses / 10).reshape(1, -1)

        pred = None
        acc = 0
        confidence = 0

        # Classical Models
        if model_choice == "SVM":
            pred, acc, confidence = run_svm(user_input)

        elif model_choice == "Logistic Regression":
            pred, acc, confidence = run_logistic(user_input)

        elif model_choice == "XGBoost":
            pred, acc, confidence = run_xgb(user_input)

        # Quantum Models
        elif model_choice == "Quantum VQC":
            df = load_data()
            if isinstance(df, tuple):
                df = df[0]
            acc = run_vqc(df)

        elif model_choice == "Quantum SVM (QSVM)":
            df = load_data()
            if isinstance(df, tuple):
                df = df[0]
            acc = run_qsvm(df)

        if pred is None:
            pred = "Quantum Model Output"
            confidence = "N/A"

        end = time.time()

    # Risk Score
    score = int(np.sum(responses))

    if score <= 25:
        label = "LOW RISK"
    elif score <= 50:
        label = "MODERATE RISK"
    elif score <= 75:
        label = "HIGH RISK"
    else:
        label = "VERY HIGH RISK"

    st.success(label)
    show_risk_gauge(score)

    # Results
    st.write("### Results")
    st.write(f"Prediction: {pred}")
    st.write(f"Confidence: {confidence}")
    st.write(f"Accuracy: {acc:.4f}")
    st.write(f"Execution Time: {end - start:.2f} sec")

    # Child Summary
    st.markdown("### Child Details Summary")
    st.write(f"Age: {age} months")
    st.write(f"Gender: {gender}")
    st.write(f"Weight: {weight} kg")
    st.write(f"Gestation: {gestation}")
    st.write(f"Family History: {family_history}")

    # ============================
    # Model Comparison
    # ============================
    st.subheader("Model Comparison")

    all_results = {}
    all_results["SVM"] = run_svm(user_input)[1]
    all_results["Logistic"] = run_logistic(user_input)[1]
    all_results["XGBoost"] = run_xgb(user_input)[1]

    df_compare = pd.DataFrame(list(all_results.items()), columns=["Model", "Accuracy"])
    st.bar_chart(df_compare.set_index("Model"))

    best_model = max(all_results, key=all_results.get)
    st.success(f"Best Performing Model: {best_model}")

    # Graph
    fig = px.bar(
        x=[f"Q{i+1}" for i in range(10)],
        y=responses,
        title="Behavioural Responses"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============================
    # Report
    # ============================
    report = {
        "Age": age,
        "Gender": gender,
        "Weight": weight,
        "Gestation": gestation,
        "Family History": family_history,
        "Model": model_choice,
        "Risk Score": score,
        "Risk Level": label,
        "Prediction": pred,
        "Confidence": confidence,
        "Accuracy": acc,
        "Generated On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # PDF
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    y = 800

    for k, v in report.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 18

    c.save()
    pdf_buffer.seek(0)

    st.download_button("Download PDF", pdf_buffer, "report.pdf")

    # Excel
    df_report = pd.DataFrame(list(report.items()), columns=["Field", "Value"])
    excel_buffer = BytesIO()
    df_report.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)

    st.download_button("Download Excel", excel_buffer, "report.xlsx")

    st.warning("This is only a screening tool, not a medical diagnosis.")

# ============================
# Footer
# ============================
st.markdown("---")

st.markdown("### About Project")
st.write("""
This project compares Classical Machine Learning and Quantum Machine Learning 
for early autism screening using behavioral data.
""")

st.markdown("### Limitations")
st.write("""
- Quantum models use small dataset due to qubit limitations
- Runs on simulator (not real quantum hardware)
- Performance depends on environment stability
""")

st.markdown("### Future Scope")
st.write("""
- Integration with real quantum hardware
- Larger dataset training
- Mobile or web deployment
""")

# 👇 YOUR NAME (FINAL TOUCH)
st.markdown(
    """
    <div style="text-align:center">
        <h3>Project Developed By</h3>
        <p><b>Akshara Addagoda</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
