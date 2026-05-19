import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Safe Report AI — WestMine",
    page_icon="⛑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main { background-color: #0f1117; }

    .stApp {
        background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #13171f;
        border-right: 1px solid #2a2f3e;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e2433 0%, #252b3b 100%);
        border: 1px solid #2a2f3e;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }

    .high-priority-card {
        background: linear-gradient(135deg, #2d1515 0%, #3d1a1a 100%);
        border: 2px solid #d62728;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
        animation: pulse-red 2s infinite;
    }

    .low-priority-card {
        background: linear-gradient(135deg, #132d1a 0%, #1a3d22 100%);
        border: 2px solid #2ca02c;
        border-radius: 12px;
        padding: 28px;
        text-align: center;
    }

    @keyframes pulse-red {
        0%, 100% { border-color: #d62728; box-shadow: 0 0 0 0 rgba(214,39,40,0.4); }
        50% { border-color: #ff4444; box-shadow: 0 0 0 8px rgba(214,39,40,0); }
    }

    .priority-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.5rem;
        font-weight: 600;
        letter-spacing: 2px;
    }

    .confidence-bar-container {
        background: #2a2f3e;
        border-radius: 8px;
        height: 12px;
        margin: 12px 0;
        overflow: hidden;
    }

    .confidence-bar-high {
        background: linear-gradient(90deg, #d62728, #ff6b6b);
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease;
    }

    .confidence-bar-low {
        background: linear-gradient(90deg, #2ca02c, #51cf66);
        height: 100%;
        border-radius: 8px;
        transition: width 0.8s ease;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #5a6478;
        margin-bottom: 8px;
    }

    .page-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #e8eaf0;
        letter-spacing: 1px;
    }

    .page-subtitle {
        color: #5a6478;
        font-size: 0.95rem;
        margin-top: 4px;
    }

    .info-box {
        background: #1e2433;
        border-left: 3px solid #4a9eff;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #8892a4;
    }

    .warning-box {
        background: #2a1f1a;
        border-left: 3px solid #ff7f0e;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #8892a4;
    }

    /* Text inputs */
    .stTextArea textarea {
        background: #1e2433 !important;
        border: 1px solid #2a2f3e !important;
        color: #e8eaf0 !important;
        border-radius: 8px !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
    }

    .stTextArea textarea:focus {
        border-color: #4a9eff !important;
        box-shadow: 0 0 0 2px rgba(74,158,255,0.15) !important;
    }

    .stNumberInput input {
        background: #1e2433 !important;
        border: 1px solid #2a2f3e !important;
        color: #e8eaf0 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1a4a8a 0%, #1e5aaa 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 32px !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1e5aaa 0%, #2266cc 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(74,158,255,0.3) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #13171f;
        border-radius: 10px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #5a6478;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }

    .stTabs [aria-selected="true"] {
        background: #1e2433;
        color: #e8eaf0;
    }

    /* Divider */
    hr { border-color: #2a2f3e; }

    /* Metric values */
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #4a9eff;
    }

    /* Sidebar nav items */
    .nav-item {
        padding: 10px 16px;
        border-radius: 8px;
        margin: 2px 0;
        cursor: pointer;
        color: #8892a4;
        font-size: 0.9rem;
        transition: all 0.2s;
    }
    .nav-item:hover { background: #1e2433; color: #e8eaf0; }
    .nav-item.active { background: #1e2433; color: #4a9eff; border-left: 3px solid #4a9eff; }
</style>
""", unsafe_allow_html=True)


# ── Helper: clean text ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


# ── Load sample data for EDA ──────────────────────────────────────────────────
@st.cache_data
def load_sample_data():
    """Load a small sample CSV for the dashboard — not the full 400MB file."""
    sample_path = os.path.join(os.path.dirname(__file__), 'data', 'dashboard_sample.csv')
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path, low_memory=False)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 28px;'>
        <div style='font-size:2.5rem;'>⛑️</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size:1.1rem;
                    font-weight:600; color:#e8eaf0; letter-spacing:2px;'>
            SAFE REPORT AI
        </div>
        <div style='font-size:0.75rem; color:#5a6478; letter-spacing:1px; margin-top:4px;'>
            WESTMINE INCIDENT CLASSIFIER
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🔍  Predict Incident", "📊  Dataset Explorer", "ℹ️  About the Model"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#5a6478; padding: 8px 0;'>
        <b style='color:#8892a4;'>MODEL</b><br>
        Logistic Regression + TF-IDF<br><br>
        <b style='color:#8892a4;'>DATASET</b><br>
        OSHA ITA Case Detail Data<br>
        2024–2025<br><br>
        <b style='color:#8892a4;'>TRAINING RECORDS</b><br>
        ~57,208 balanced records<br><br>
        <b style='color:#8892a4;'>TEST AUC-ROC</b><br>
        0.7323
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "Predict" in page:
    st.markdown("""
    <div class='page-title'>⛑️ Incident Report Classifier</div>
    <div class='page-subtitle'>Enter the incident details below to classify priority level</div>
    <br>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model file not found. Make sure `models/best_model.joblib` exists.")
        st.stop()

    st.markdown("<div class='section-header'>Narrative Fields</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Fill in as many fields as possible. The more detail provided, the more accurate the prediction.
        Fields marked with * are most important for classification.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        desc = st.text_area(
            "📋 Incident Description *",
            placeholder="e.g. Employee slipped on wet floor near loading dock",
            height=100,
            help="A brief description of the incident"
        )
        what_happened = st.text_area(
            "❗ What Happened *",
            placeholder="e.g. Worker was walking to the break room when they slipped on a wet floor and fell, striking their head on a metal shelf",
            height=120,
            help="Detailed account of how the incident occurred"
        )
        illness = st.text_area(
            "🩺 Injury / Illness Description *",
            placeholder="e.g. Laceration to forehead requiring 8 stitches, possible concussion",
            height=100,
            help="Description of the resulting injury or illness"
        )

    with col2:
        before = st.text_area(
            "⏪ What Was Employee Doing Before?",
            placeholder="e.g. Walking from workstation to break room at end of shift",
            height=100,
            help="What was the employee doing immediately before the incident"
        )
        obj_substance = st.text_area(
            "🔧 Object / Substance Involved",
            placeholder="e.g. Wet floor, metal shelf",
            height=100,
            help="Object or substance that directly caused harm"
        )
        location = st.text_area(
            "📍 Incident Location",
            placeholder="e.g. Warehouse floor near loading dock",
            height=80,
            help="Where the incident occurred"
        )

    st.markdown("<br><div class='section-header'>Severity Indicators</div>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        dafw = st.number_input("Days Away From Work", min_value=0, max_value=365, value=0,
                               help="Number of days the employee was absent")
    with col4:
        djtr = st.number_input("Days Job Transfer / Restriction", min_value=0, max_value=365, value=0,
                               help="Number of days on restricted duty")
    with col5:
        total_days = dafw + djtr
        st.metric("Total Severity Days", total_days)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  CLASSIFY INCIDENT REPORT"):
        # Check at least one narrative field is filled
        if not any([desc, what_happened, illness, before, obj_substance, location]):
            st.warning("Please fill in at least one narrative field before classifying.")
        else:
            # Build input dataframe with cleaned text
            input_df = pd.DataFrame([{
                'New_incident_description': clean_text(desc),
                'New_nar_before_incident':  clean_text(before),
                'New_nar_what_happened':    clean_text(what_happened),
                'New_nar_injury_illness':   clean_text(illness),
                'New_nar_object_substance': clean_text(obj_substance),
                'New_incident_location':    clean_text(location),
                'dafw_clipped':             min(dafw, 180),
                'djtr_clipped':             min(djtr, 180),
                'total_severity_days':      min(dafw + djtr, 360),
            }])

            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]
            high_prob = probability[1]
            low_prob  = probability[0]

            st.markdown("<hr><div class='section-header'>Classification Result</div>", unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(f"""
                <div class='high-priority-card'>
                    <div class='priority-label' style='color:#ff4444;'>🚨 HIGH PRIORITY</div>
                    <div style='color:#ff8888; margin-top:8px; font-size:0.95rem;'>
                        This incident requires immediate attention and escalation.
                    </div>
                    <br>
                    <div style='color:#8892a4; font-size:0.85rem; font-family: IBM Plex Mono, monospace;'>
                        CONFIDENCE
                    </div>
                    <div class='confidence-bar-container'>
                        <div class='confidence-bar-high' style='width:{high_prob*100:.1f}%'></div>
                    </div>
                    <div style='color:#ff4444; font-size:1.4rem; font-family: IBM Plex Mono, monospace; font-weight:600;'>
                        {high_prob*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='low-priority-card'>
                    <div class='priority-label' style='color:#51cf66;'>✅ LOW PRIORITY</div>
                    <div style='color:#88cc88; margin-top:8px; font-size:0.95rem;'>
                        This incident is recordable but does not require immediate escalation.
                    </div>
                    <br>
                    <div style='color:#8892a4; font-size:0.85rem; font-family: IBM Plex Mono, monospace;'>
                        CONFIDENCE
                    </div>
                    <div class='confidence-bar-container'>
                        <div class='confidence-bar-low' style='width:{low_prob*100:.1f}%'></div>
                    </div>
                    <div style='color:#51cf66; font-size:1.4rem; font-family: IBM Plex Mono, monospace; font-weight:600;'>
                        {low_prob*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            col6, col7 = st.columns(2)
            with col6:
                st.metric("High Priority Probability", f"{high_prob*100:.1f}%")
            with col7:
                st.metric("Low Priority Probability", f"{low_prob*100:.1f}%")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=high_prob * 100,
                title={'text': "High Priority Score", 'font': {'color': '#8892a4', 'size': 14}},
                number={'suffix': "%", 'font': {'color': '#e8eaf0', 'size': 28}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#5a6478'},
                    'bar': {'color': '#d62728' if prediction == 1 else '#2ca02c'},
                    'bgcolor': '#1e2433',
                    'bordercolor': '#2a2f3e',
                    'steps': [
                        {'range': [0, 50],  'color': '#132d1a'},
                        {'range': [50, 100], 'color': '#2d1515'},
                    ],
                    'threshold': {
                        'line': {'color': '#ffffff', 'width': 2},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(
                height=280,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#8892a4'},
                margin=dict(t=40, b=0, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""<div class='warning-box'>
                ⚠️ This classification is AI-assisted and should be reviewed by a safety officer
                before final determination. Model AUC-ROC: 0.7323.
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif "Explorer" in page:
    st.markdown("""
    <div class='page-title'>📊 Dataset Explorer</div>
    <div class='page-subtitle'>Explore patterns in the OSHA ITA incident dataset</div>
    <br>
    """, unsafe_allow_html=True)

    df = load_sample_data()

    if df is None:
        st.markdown("""<div class='warning-box'>
            Dashboard sample data not found. To enable this page:<br>
            1. Run your Jupyter notebook to generate the balanced dataset<br>
            2. Save a sample: <code>df_balanced.sample(5000, random_state=42).to_csv('data/dashboard_sample.csv', index=False)</code><br>
            3. Place it in the <code>data/</code> folder
        </div>""", unsafe_allow_html=True)

        # Show static charts as fallback using the known values from your results
        st.markdown("### Dataset Overview (from model training)")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Full Dataset", "688,650 records")
        col2.metric("Training Sample", "100,000 records")
        col3.metric("Balanced Dataset", "~71,510 records")
        col4.metric("Test Set", "14,302 records")

        st.markdown("<br>", unsafe_allow_html=True)

        # Static outcome distribution from your actual results
        fig1 = go.Figure(data=[
            go.Bar(
                x=['Days Away From Work', 'Other Recordable', 'Job Transfer/Restriction', 'Death'],
                y=[35726, 34033, 30211, 29],
                marker_color=['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728'],
                text=[35726, 34033, 30211, 29],
                textposition='outside',
                textfont={'color': '#e8eaf0'}
            )
        ])
        fig1.update_layout(
            title='Incident Outcome Distribution (100k Sample)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#8892a4'},
            xaxis={'gridcolor': '#2a2f3e', 'tickfont': {'color': '#8892a4'}},
            yaxis={'gridcolor': '#2a2f3e', 'tickfont': {'color': '#8892a4'}},
            title_font={'color': '#e8eaf0'},
            height=380
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Incident type vs priority from your actual data
        fig2 = go.Figure(data=[
            go.Bar(name='High Priority', x=['Respiratory', 'Poisoning', 'Other Illness', 'Injury', 'Skin Disorder', 'Hearing Loss'],
                   y=[82.4, 27.3, 33.3, 35.2, 25.9, 9.5],
                   marker_color='#d62728', opacity=0.85),
            go.Bar(name='Low Priority',  x=['Respiratory', 'Poisoning', 'Other Illness', 'Injury', 'Skin Disorder', 'Hearing Loss'],
                   y=[17.6, 72.7, 66.7, 64.8, 74.1, 90.5],
                   marker_color='#1f77b4', opacity=0.85),
        ])
        fig2.update_layout(
            barmode='stack',
            title='Incident Type vs Priority Label (% composition)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#8892a4'},
            xaxis={'gridcolor': '#2a2f3e', 'tickfont': {'color': '#8892a4'}},
            yaxis={'gridcolor': '#2a2f3e', 'tickfont': {'color': '#8892a4'},
                   'title': 'Percentage (%)', 'range': [0, 100]},
            title_font={'color': '#e8eaf0'},
            legend={'font': {'color': '#8892a4'}},
            height=380
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Model performance summary
        st.markdown("### Model Performance on Unseen Test Set")
        col5, col6 = st.columns(2)

        with col5:
            perf_lr = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 (weighted)', 'Recall — High Priority',
                           'Precision — High Priority', 'AUC-ROC'],
                'Score': [0.6601, 0.6599, 0.6882, 0.6516, 0.7323]
            })
            fig3 = go.Figure(go.Bar(
                x=perf_lr['Score'], y=perf_lr['Metric'],
                orientation='h', marker_color='#1f77b4',
                text=[f"{v:.4f}" for v in perf_lr['Score']],
                textposition='outside', textfont={'color': '#e8eaf0'}
            ))
            fig3.update_layout(
                title='Logistic Regression',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#8892a4'},
                xaxis={'range': [0, 1], 'gridcolor': '#2a2f3e',
                       'tickfont': {'color': '#8892a4'}},
                yaxis={'tickfont': {'color': '#8892a4'}},
                title_font={'color': '#e8eaf0'},
                height=320
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col6:
            perf_rf = pd.DataFrame({
                'Metric': ['Accuracy', 'F1 (weighted)', 'Recall — High Priority',
                           'Precision — High Priority', 'AUC-ROC'],
                'Score': [0.6373, 0.6215, 0.8420, 0.5974, 0.7101]
            })
            fig4 = go.Figure(go.Bar(
                x=perf_rf['Score'], y=perf_rf['Metric'],
                orientation='h', marker_color='#ff7f0e',
                text=[f"{v:.4f}" for v in perf_rf['Score']],
                textposition='outside', textfont={'color': '#e8eaf0'}
            ))
            fig4.update_layout(
                title='Random Forest',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#8892a4'},
                xaxis={'range': [0, 1], 'gridcolor': '#2a2f3e',
                       'tickfont': {'color': '#8892a4'}},
                yaxis={'tickfont': {'color': '#8892a4'}},
                title_font={'color': '#e8eaf0'},
                height=320
            )
            st.plotly_chart(fig4, use_container_width=True)

    else:
        # ── Live charts from actual data ──────────────────────────────────────
        # Map labels if needed
        if 'incident_outcome' in df.columns:
            df['incident_outcome'] = df['incident_outcome'].astype(int)
            priority_map = {1: 'High', 2: 'High', 3: 'Low', 4: 'Low'}
            df['priority_label_str'] = df['incident_outcome'].map(priority_map)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Records in Sample", f"{len(df):,}")
        if 'priority_label_str' in df.columns:
            high_count = (df['priority_label_str'] == 'High').sum()
            low_count  = (df['priority_label_str'] == 'Low').sum()
            col2.metric("High Priority", f"{high_count:,}")
            col3.metric("Low Priority",  f"{low_count:,}")
            col4.metric("High Priority %", f"{high_count/len(df)*100:.1f}%")

        tab1, tab2, tab3 = st.tabs(["Outcome Distribution", "Incident Types", "Geographic"])

        with tab1:
            if 'incident_outcome' in df.columns:
                outcome_labels = {1: 'Death', 2: 'Days Away From Work',
                                  3: 'Job Transfer/Restriction', 4: 'Other Recordable'}
                outcome_colors = {1: '#d62728', 2: '#ff7f0e', 3: '#2ca02c', 4: '#1f77b4'}
                counts = df['incident_outcome'].value_counts().sort_index()
                fig = go.Figure(data=[
                    go.Bar(
                        x=[outcome_labels.get(i, str(i)) for i in counts.index],
                        y=counts.values,
                        marker_color=[outcome_colors.get(i, '#888') for i in counts.index],
                        text=counts.values,
                        textposition='outside',
                        textfont={'color': '#e8eaf0'}
                    )
                ])
                fig.update_layout(
                    title='Incident Outcome Distribution',
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#8892a4'},
                    xaxis={'gridcolor': '#2a2f3e'},
                    yaxis={'gridcolor': '#2a2f3e'},
                    title_font={'color': '#e8eaf0'},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            if 'type_of_incident' in df.columns:
                type_labels = {1:'Injury', 2:'Skin Disorder', 3:'Respiratory',
                               4:'Poisoning', 5:'Hearing Loss', 6:'Other Illness'}
                df['type_label'] = df['type_of_incident'].map(type_labels)
                type_counts = df['type_label'].value_counts()
                fig = px.bar(x=type_counts.values, y=type_counts.index,
                             orientation='h',
                             color=type_counts.values,
                             color_continuous_scale='Blues',
                             title='Type of Incident')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#8892a4'},
                    title_font={'color': '#e8eaf0'},
                    coloraxis_showscale=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            if 'state' in df.columns:
                state_counts = df['state'].value_counts().head(15)
                fig = px.bar(x=state_counts.index, y=state_counts.values,
                             title='Top 15 States by Incident Count',
                             color=state_counts.values,
                             color_continuous_scale='Blues')
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#8892a4'},
                    title_font={'color': '#e8eaf0'},
                    coloraxis_showscale=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT THE MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    <div class='page-title'>ℹ️ About the Model</div>
    <div class='page-subtitle'>How Safe Report AI works</div>
    <br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Problem Statement")
        st.markdown("""
        <div class='metric-card'>
        WestMine receives hundreds of workplace incident reports and needs to quickly identify
        which ones require immediate escalation versus routine follow-up.
        <br><br>
        Manual review of every report is time-consuming. Safe Report AI automates the
        initial triage by classifying each report as <b style='color:#ff4444'>High Priority</b>
        (death or days away from work) or <b style='color:#51cf66'>Low Priority</b>
        (job transfer or other recordable).
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Dataset")
        st.markdown("""
        <div class='metric-card'>
        <b>Source:</b> OSHA Injury Tracking Application (ITA) Case Detail Data 2024–2025<br>
        <b>Full dataset:</b> 688,650 records<br>
        <b>Training sample:</b> 100,000 (stratified)<br>
        <b>After balancing:</b> ~71,510 records (1:1 High/Low)<br>
        <b>Train/Test split:</b> 80% / 20% (stratified, seeded)
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Methodology")
        st.markdown("""
        <div class='metric-card'>
        <b>Type:</b> NLP-based Binary Classification<br><br>
        <b>Input features:</b><br>
        &nbsp;&nbsp;• 6 narrative text fields (separate TF-IDF each)<br>
        &nbsp;&nbsp;• Days Away From Work (clipped at 99th percentile)<br>
        &nbsp;&nbsp;• Days Job Transfer/Restriction<br>
        &nbsp;&nbsp;• Total Severity Days<br><br>
        <b>Models compared:</b><br>
        &nbsp;&nbsp;• Logistic Regression + ColumnTransformer (selected)<br>
        &nbsp;&nbsp;• Random Forest + ColumnTransformer<br><br>
        <b>Selection criterion:</b> AUC-ROC on unseen test set
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Performance")
        st.markdown("""
        <div class='metric-card'>
        <b>Best Model:</b> Logistic Regression<br><br>

        | Metric | Score |
        |--------|-------|
        | Accuracy | 66.0% |
        | AUC-ROC | 0.7323 |
        | Recall — High Priority | 68.8% |
        | Precision — High Priority | 65.2% |
        | F1 (weighted) | 66.0% |
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Target Variable")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
        <div class='high-priority-card'>
            <div style='font-size:1.5rem; font-weight:700; color:#ff4444;
                        font-family: IBM Plex Mono, monospace;'>HIGH PRIORITY</div>
            <div style='color:#cc8888; margin-top:12px;'>
                ⚰️ Death (Outcome Code 1)<br>
                🏥 Days Away From Work (Outcome Code 2)
            </div>
            <div style='color:#5a6478; font-size:0.85rem; margin-top:12px;'>
                Requires immediate escalation and review
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='low-priority-card'>
            <div style='font-size:1.5rem; font-weight:700; color:#51cf66;
                        font-family: IBM Plex Mono, monospace;'>LOW PRIORITY</div>
            <div style='color:#88cc88; margin-top:12px;'>
                🔄 Job Transfer / Restriction (Outcome Code 3)<br>
                📋 Other Recordable Case (Outcome Code 4)
            </div>
            <div style='color:#5a6478; font-size:0.85rem; margin-top:12px;'>
                Recordable but does not require immediate escalation
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class='warning-box'>
        ⚠️ <b>Disclaimer:</b> Safe Report AI is an AI-assisted triage tool.
        All classifications should be reviewed by a qualified safety officer before
        final determination. The model achieves AUC-ROC of 0.7323 on unseen data —
        it is not infallible. Missing a High Priority incident is more dangerous than
        a false alarm, which is why Recall is prioritised over Precision.
    </div>""", unsafe_allow_html=True)
