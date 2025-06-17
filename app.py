import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

# Load model and encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="AI Disease Predictor", layout="wide")
st.markdown("""
<style>
body, .stApp {
    background-color: #0e1117;
    color: #d0d0d0;
    font-family: 'Segoe UI', sans-serif;
}
input, select {
    background-color: #1e1e1e !important;
    color: #d0d0d0 !important;
}
.stButton > button {
    background-color: #4CAF50 !important;
    color: white !important;
    border-radius: 8px;
    font-weight: 600;
    height: 42px;
}
.stNumberInput input {
    background-color: #1e1e1e !important;
}
.result-box {
    background-color: #1f2937;
    padding: 1.5em;
    border-radius: 12px;
    margin-top: 1.5em;
    text-align: center;
}
.result-box h3 {
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 AI Disease Prediction Tool")
st.markdown("Provide patient health parameters below to predict possible disease.")

# Input form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 1, 120, 30)
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)

    with col2:
        bp = st.number_input("Blood Pressure (mmHg)", 80, 200, 120)
        sugar = st.number_input("Sugar Level (mg/dL)", 50, 300, 100)
        smoking = st.selectbox("Smoking Status", encoders["Smoking"].classes_)

    with col3:
        chol = st.number_input("Cholesterol (mg/dL)", 100, 400, 180)
        fam = st.selectbox("Family History", encoders["FamilyHistory"].classes_)

    submitted = st.form_submit_button("🔍 Predict Disease")

# On form submission
if submitted:
    # Prepare input
    input_dict = {
        "Age": age,
        "Gender": encoders["Gender"].transform([gender])[0],
        "BMI": bmi,
        "BP": bp,
        "Sugar": sugar,
        "Cholesterol": chol,
        "Smoking": encoders["Smoking"].transform([smoking])[0],
        "FamilyHistory": encoders["FamilyHistory"].transform([fam])[0]
    }
    X = pd.DataFrame([input_dict])

    # Predict
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X)[0]
    disease_classes = encoders["Disease"].inverse_transform(model.classes_)

    predicted_disease = encoders["Disease"].inverse_transform([prediction])[0]
    accuracy= prediction_proba[prediction] * 100

    # Top 3 predictions
    top3_idx = np.argsort(prediction_proba)[::-1][:3]
    top3_diseases = disease_classes[top3_idx]
    top3_probs = prediction_proba[top3_idx] * 100

    # Display result
    st.markdown(f"""
    <div class='result-box'>
        <h3>🎯 Predicted Disease: {predicted_disease}</h3>
        <p><strong>Accuracy:</strong> {accuracy:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # Optional visualization
    fig = px.bar(x=top3_diseases, y=top3_probs,
                 labels={'x': 'Disease', 'y': 'Chances (%)'},
                 color=top3_diseases,
                 color_discrete_sequence=px.colors.sequential.Viridis)
    fig.update_layout(title="Chances(%)", height=350, xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    # Recommendation
    st.markdown("### 💡 Recommendation")
    if predicted_disease == "No Disease":
        st.success("You're doing great! Stay healthy 🧘‍♂️")
    else:
        st.warning(f"Consult a physician regarding **{predicted_disease}**.")
        st.markdown("""
        - 🥦 Maintain a balanced diet  
        - 🚶‍♀️ Stay physically active  
        - 🛌 Get 7–8 hours of sleep  
        - 🚭 Quit smoking/alcohol (if applicable)  
        - 🧪 Monitor BP, Sugar, Cholesterol regularly  
        """)