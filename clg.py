import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("admission_score_data.csv")

X = df[["12th Score (%)", "Number of Extracurriculars", "Entrance Exam Score", 
        "Olympiads Participated", "Volunteer Hours"]]
y = df["Admission Score (0-100)"]

model = LinearRegression()
model.fit(X, y)

st.set_page_config(page_title="🎓 Path2College - Admission Score Predictor", layout="centered")

st.markdown("""
    <style>
    body {
        background-color: #f0f4f8;
    }
    .main-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-top: 1rem;
    }
    h1, h2, h3 {
        color: #2b547e;
    }
    .stSlider label, .stNumberInput label {
        color: #333333;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #2b547e;
        color: white;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("🎓 Path2College – Admission Score Predictor")
st.markdown("🔍 Estimate your **college admission score (0–100)** based on academic & extracurricular achievements.")

col1, col2 = st.columns(2)
with col1:
    twelve_score = st.slider("📘 12th Score (%)", 60, 100, 75)
    olympiads = st.slider("🥇 Olympiads Participated", 0, 5, 1)
    volunteer = st.slider("🤝 Volunteer Hours", 0, 100, 20)
with col2:
    extra = st.slider("🎨 No. of Extracurriculars", 0, 10, 3)
    exam = st.slider("📝 Entrance Exam Score", 100, 200, 150)

input_data = [[twelve_score, extra, exam, olympiads, volunteer]]
predicted = model.predict(input_data)[0]

st.markdown("---")
st.success(f"🎯 Your predicted **admission score** is: **{predicted:.2f} / 100**")

st.markdown("### 📈 Feature Importance (Higher is More Influential)")
st.bar_chart(pd.Series(model.coef_, index=X.columns).sort_values(ascending=False))

with st.expander("📁 View Sample Data"):
    st.dataframe(df.head())

with st.expander("⬇️ Download Dataset"):
    with open("admission_score_data.csv", "rb") as f:
        st.download_button("Download CSV", f, file_name="admission_score_data.csv", mime="text/csv")

st.markdown('</div>', unsafe_allow_html=True)
