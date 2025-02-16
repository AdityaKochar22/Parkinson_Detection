import streamlit as st
import numpy as np
import joblib
import time
import plotly.graph_objects as go

# Set Page Config
st.set_page_config(page_title="🧠 Parkinson's Prediction", layout="wide")

# Load trained model and scaler
model = joblib.load("parkinsons_model.pkl")
scaler = joblib.load("scaler.pkl")

# 🏆 Animated Header
st.markdown("""
    <h1 style='text-align: center; font-size: 36px; color: #4A90E2; animation: fadeIn 1.5s ease-in;'>
    🧠 Parkinson's Disease Prediction System</h1>
    <style>
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
""", unsafe_allow_html=True)

st.write("Enter feature values below for prediction.")

# 🎛️ User Input Fields (No Sliders)
col1, col2 = st.columns(2)
with col1:
    RPDE = st.number_input("🔹 RPDE (Recurrence Period Density Entropy)", min_value=0.0, max_value=1.0, format="%.5f")
    DFA = st.number_input("🔹 DFA (Detrended Fluctuation Analysis)", min_value=0.0, max_value=3.0, format="%.5f")
    PPE = st.number_input("🔹 PPE (Pitch Period Entropy)", min_value=0.0, max_value=1.0, format="%.5f")

with col2:
    HNR = st.number_input("🔹 HNR (Harmonic-to-Noise Ratio)", min_value=0.0, max_value=50.0, format="%.2f")
    Shimmer_APQ5 = st.number_input("🔹 Shimmer:APQ5", min_value=0.0, max_value=1.0, format="%.5f")

# 🔹 Add Space After Inputs
st.markdown("<br><br>", unsafe_allow_html=True)

# 🚀 Centered Predict Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🚀 Predict", use_container_width=True)

# 🔹 Add Space Before Prediction Results
st.markdown("<br>", unsafe_allow_html=True)

if predict_button:
    with st.spinner("🔄 Processing... Please wait."):
        progress_bar = st.progress(0)
        
        # Simulate Progress
        for i in range(100):
            time.sleep(0.005)  # Faster progress bar
            progress_bar.progress(i + 1)

        # Prepare input data
        input_data = np.array([[RPDE, DFA, HNR, Shimmer_APQ5, PPE]])
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100  # Convert to percentage

    # ✅ Dynamic Styling Based on Prediction
    if prediction == 1:  # High Risk (Parkinson's Likely)
        result_message = "⚠️ High Risk: The patient is likely to have Parkinson’s Disease."
        bg_color = "#F44336"  # Red
        pulse_animation = "pulseRed"
    else:  # Low Risk (Healthy)
        result_message = "✅ Low Risk: The patient is unlikely to have Parkinson’s Disease."
        bg_color = "#4CAF50"  # Green
        pulse_animation = "pulseGreen"

    # 📌 Styled Output Box with Pulse Effect
    st.markdown(f"""
        <style>
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        @keyframes {pulse_animation} {{
            0% {{ box-shadow: 0 0 10px {bg_color}; }}
            50% {{ box-shadow: 0 0 20px {bg_color}; }}
            100% {{ box-shadow: 0 0 10px {bg_color}; }}
        }}
        .prediction-container {{
            background-color: {bg_color};
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 18px;
            color: white;
            animation: fadeIn 1s ease-in, {pulse_animation} 2s infinite alternate;
        }}
        </style>
        <div class='prediction-container'>{result_message}</div>
    """, unsafe_allow_html=True)

    # 🔹 Add Space Before Gauge Chart
    st.markdown("<br>", unsafe_allow_html=True)

    # 📊 Centered Gauge Chart
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:  # Center column for the gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability,
            title={"text": "Prediction Confidence (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": 'teal'},  # Bar color remains black
                "steps": [
                    {"range": [0, 50], "color": "rgba(76, 175, 80, 0.8)"},  # Light Green for low risk
                    {"range": [50, 80], "color": "rgba(255, 165, 0, 0.8)"},  # Orange for medium risk
                    {"range": [80, 100], "color": "rgba(255, 76, 76, 0.9)"}  # Deep Red for high risk
                ],
                "borderwidth": 2,
                "bordercolor": "#444"
            }
        ))
        st.plotly_chart(fig)
