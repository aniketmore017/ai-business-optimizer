import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import requests

st.markdown("""
<style>

/* ---------------- MAIN BACKGROUND ---------------- */
.stApp {
    background: #f5f7fb;
}

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"] {
    background: #ffffff;
}

/* ---------------- KPI CARDS ---------------- */
[data-testid="metric-container"] {
    background: white;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
}

/* ---------------- BUTTON ---------------- */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

/* ---------------- FILE UPLOADER ---------------- */
[data-testid="stFileUploader"] {
    background-color: white;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #ddd;
}

/* ---------------- DROPDOWN ---------------- */
div[data-baseweb="select"] {
    background-color: white;
    border-radius: 8px;
}

/* ---------------- TABLE ---------------- */
thead tr th {
    background-color: #f0f2f6;
    color: black;
}

tbody tr td {
    background-color: white;
    color: black;
}

/* ---------------- TEXT FIX ---------------- */
h1, h2, h3, h4, h5, h6, p, label {
    color: black;
}

/* ---------------- CODE TEXT ---------------- */
code {
    background-color: #e8f0fe;
    color: #0b5394;
    padding: 3px 6px;
    border-radius: 6px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
API_KEY = "AIzaSyDpvyDsHl5auXZBRBzuQSBXoX0hX5IlLOY"

st.set_page_config(page_title="AI Business Optimizer", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Filters")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center;'>AI Business Process Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Optimize operations using ML + AI insights</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ---------------- SIDEBAR FILTER ----------------
    st.sidebar.header("🔍 Filters")

# -------- REGION FILTER --------
    region = st.sidebar.multiselect(
        "Select Region",
        options=df['region'].unique(),
        default=df['region'].unique()
    )

    # -------- STATUS FILTER --------
    status = st.sidebar.multiselect(
        "Select Status",
        options=df['status'].unique(),
        default=df['status'].unique()
    )

    # -------- ERROR FLAG FILTER --------
    error_flag = st.sidebar.multiselect(
        "Error Flag",
        options=df['error_flag'].unique(),
        default=df['error_flag'].unique()
    )

    # -------- PROCESSING TIME FILTER --------
    min_proc, max_proc = st.sidebar.slider(
        "Processing Time Range",
        int(df['processing_time'].min()),
        int(df['processing_time'].max()),
        (int(df['processing_time'].min()), int(df['processing_time'].max()))
    )

    # -------- DELIVERY TIME FILTER --------
    min_del, max_del = st.sidebar.slider(
        "Delivery Time Range",
        int(df['delivery_time'].min()),
        int(df['delivery_time'].max()),
        (int(df['delivery_time'].min()), int(df['delivery_time'].max()))
    )

    # -------- APPLY FILTERS --------
    df = df[
        (df['region'].isin(region)) &
        (df['status'].isin(status)) &
        (df['error_flag'].isin(error_flag)) &
        (df['processing_time'].between(min_proc, max_proc)) &
        (df['delivery_time'].between(min_del, max_del))
    ]

    # ---------------- KPI CARDS ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric("📦 Total Orders", len(df))
    col2.metric("⏱ Avg Processing Time", round(df['processing_time'].mean(), 2))
    col3.metric("🚚 Avg Delivery Time", round(df['delivery_time'].mean(), 2))

    # ---------------- DATA ----------------
    st.subheader("📊 Data Preview")
    st.dataframe(df)

    # ---------------- ML MODEL ----------------
    model = IsolationForest(contamination=0.25, random_state=42)
    df['anomaly'] = model.fit_predict(df[['processing_time', 'delivery_time']])

    anomalies = df[df['anomaly'] == -1]

    st.subheader("🚨 Detected Anomalies")
    st.dataframe(anomalies)

    # ---------------- CHART ----------------
    st.subheader("📈 Processing vs Delivery Time")

    fig, ax = plt.subplots(figsize=(4, 3))  # width, height

    ax.scatter(df['processing_time'], df['delivery_time'], c=df['anomaly'])
    ax.set_xlabel("Processing Time")
    ax.set_ylabel("Delivery Time")

    st.pyplot(fig)

    # ---------------- AI BUTTON ----------------
    if st.button("🤖 Generate AI Insights"):

        with st.spinner("Analyzing data with AI..."):

            summary = df.describe().to_string()
            problem_data = anomalies.to_string()

            prompt = f"""
            You are a business expert.

            Dataset Summary:
            {summary}

            Problematic Orders:
            {problem_data}

            Tell:
            1. Why delays are happening
            2. Which region has problem
            3. How to improve process
            """

            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={API_KEY}"

            headers = {"Content-Type": "application/json"}

            data = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }

            response = requests.post(url, headers=headers, json=data)
            result = response.json()

            if 'candidates' in result:
                insights = result['candidates'][0]['content']['parts'][0]['text']
                st.subheader("📊 AI Insights")
                st.write(insights)
            else:
                st.error(f"Error: {result}")