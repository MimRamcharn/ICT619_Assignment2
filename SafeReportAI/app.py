import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
import plotly.express as px
import plotly.graph_objects as go

#Page config 
st.set_page_config(
    page_title="Safe Report AI — WestMine",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Custom CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1f2e 100%); }

    section[data-testid="stSidebar"] {
        background: #13171f;
        border-right: 1px solid #2a2f3e;
    }

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
        50%       { border-color: #ff4444; box-shadow: 0 0 0 8px rgba(214,39,40,0); }
    }

    .priority-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        letter-spacing: 2px;
    }

    .confidence-bar-container {
        background: #2a2f3e;
        border-radius: 8px;
        height: 14px;
        margin: 14px 0;
        overflow: hidden;
    }

    .confidence-bar-high {
        background: linear-gradient(90deg, #d62728, #ff6b6b);
        height: 100%;
        border-radius: 8px;
    }

    .confidence-bar-low {
        background: linear-gradient(90deg, #2ca02c, #51cf66);
        height: 100%;
        border-radius: 8px;
    }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #5a6478;
        margin-bottom: 8px;
    }

    .page-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.7rem;
        font-weight: 600;
        color: #e8eaf0;
        letter-spacing: 1px;
    }

    .page-subtitle {
        color: #5a6478;
        font-size: 0.93rem;
        margin-top: 4px;
    }

    .info-box {
        background: #1e2433;
        border-left: 3px solid #4a9eff;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 12px 0;
        font-size: 0.88rem;
        color: #8892a4;
    }

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
        width: 100% !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1e5aaa 0%, #2266cc 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(74,158,255,0.3) !important;
    }

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

    hr { border-color: #2a2f3e; }

    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        color: #4a9eff;
    }

    /* Classification result table */
    .result-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.88rem;
        margin-top: 12px;
    }
    .result-table th {
        background: #1e2433;
        color: #5a6478;
        padding: 10px 14px;
        text-align: left;
        font-size: 0.72rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        border-bottom: 1px solid #2a2f3e;
    }
    .result-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #1e2433;
        color: #e8eaf0;
    }
    .result-table tr:hover td { background: #1e2433; }
    .val-high  { color: #ff4444; font-weight: 600; }
    .val-low   { color: #51cf66; font-weight: 600; }
    .val-blue  { color: #4a9eff; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


#Helpers
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_sample_data():
    path = os.path.join(os.path.dirname(__file__), 'data', 'dashboard_sample.csv')
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return None

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font={'color': '#8892a4'},
    title_font={'color': '#e8eaf0', 'size': 14},
    xaxis={'gridcolor': '#2a2f3e', 'tickfont': {'color': '#8892a4'}},
    yaxis={'gridcolor': '#2a2f3e', 'tickfont': {'color': '#8892a4'}},
    legend={'font': {'color': '#8892a4'}},
    height=380,
    margin=dict(t=48, b=20, l=20, r=20)
)


#Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 24px;'>
        <div style='font-family:IBM Plex Mono,monospace; font-size:1.05rem;
                    font-weight:600; color:#e8eaf0; letter-spacing:2px;'>
            SAFE REPORT AI
        </div>
        <div style='font-size:0.72rem; color:#5a6478; letter-spacing:1px; margin-top:4px;'>
            WESTMINE INCIDENT CLASSIFIER
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["About the Model", "Dataset Explorer", "Predict Incident"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#5a6478; padding:8px 0; line-height:1.8;'>
        <b style='color:#8892a4;'>MODEL</b><br>
        Logistic Regression + TF-IDF<br><br>
        <b style='color:#8892a4;'>DATASET</b><br>
        OSHA ITA Case Detail Data 2024–2025<br><br>
        <b style='color:#8892a4;'>TRAINING RECORDS</b><br>
        ~57,208 balanced records<br><br>
        <b style='color:#8892a4;'>TEST AUC-ROC</b><br>
        0.7323
    </div>
    """, unsafe_allow_html=True)



# PAGE 1 — ABOUT THE MODEL
# ══════════════════════════════════════════════════════════════════════════════
if page == "About the Model":
    st.markdown("""
    <div class='page-title'>About the Model</div>
    <div class='page-subtitle'>How Safe Report AI works and what it was built on</div>
    <br>
    """, unsafe_allow_html=True)

    #Problem
    st.markdown("### Problem Statement")
    st.markdown("""
    <div class='metric-card'>
    WestMine receives a large volume of workplace incident reports and needs to quickly
    identify which ones require immediate escalation versus routine follow-up.<br><br>
    Manual review of every report is time-consuming and prone to delays. Safe Report AI
    automates the initial triage by reading the incident narrative and classifying each
    report as <b style='color:#ff4444'>High Priority</b> or
    <b style='color:#51cf66'>Low Priority</b>.
    </div>
    """, unsafe_allow_html=True)

    #Target variable
    st.markdown("### Target Variable — Priority Label")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='high-priority-card'>
            <div class='priority-label' style='color:#ff4444;'>HIGH PRIORITY</div>
            <div style='color:#cc8888; margin-top:14px; line-height:1.9;'>
                Death (Outcome Code 1)<br>
                Days Away From Work (Outcome Code 2)
            </div>
            <div style='color:#5a6478; font-size:0.83rem; margin-top:12px;'>
                Requires immediate escalation and safety officer review
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='low-priority-card'>
            <div class='priority-label' style='color:#51cf66;'>LOW PRIORITY</div>
            <div style='color:#88cc88; margin-top:14px; line-height:1.9;'>
                Job Transfer / Restriction (Outcome Code 3)<br>
                Other Recordable Case (Outcome Code 4)
            </div>
            <div style='color:#5a6478; font-size:0.83rem; margin-top:12px;'>
                Recordable but does not require immediate escalation
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #Dataset 
    st.markdown("### Dataset")
    st.markdown("""
    <div class='metric-card'>
        <b>Source:</b> OSHA Injury Tracking Application (ITA) Case Detail Data 2024–2025<br><br>
        <b>Full dataset:</b> 688,650 records across 39 columns<br>
        <b>Stratified sample:</b> 100,000 records (proportional per outcome class)<br>
        <b>After downsampling to balance classes 1:1:</b> ~71,510 records<br>
        <b>Training set (80%):</b> ~57,208 records<br>
        <b>Test set (20% — unseen):</b> 14,302 records<br><br>
        <b>Class balance:</b> Equal High Priority and Low Priority records in both splits.
        Downsampling was used to prevent the model from being biased toward the majority class.
    </div>
    """, unsafe_allow_html=True)

    #Methodology
    st.markdown("### Methodology")
    st.markdown("""
    <div class='metric-card'>
        <b>Type:</b> NLP-based Binary Text Classification<br><br>
        <b>Input features (9 total):</b><br>
        &nbsp;&nbsp;&nbsp;6 narrative text fields — each processed by its own TF-IDF vectoriser<br>
        &nbsp;&nbsp;&nbsp;Days Away From Work (clipped at 99th percentile)<br>
        &nbsp;&nbsp;&nbsp;Days Job Transfer / Restriction (clipped at 99th percentile)<br>
        &nbsp;&nbsp;&nbsp;Total Severity Days (sum of above two)<br><br>
        <b>Why separate TF-IDF per field?</b><br>
        Each narrative field serves a different purpose. Giving each its own vectoriser
        means location words (e.g. "warehouse") do not compete with injury words
        (e.g. "fracture") for the same feature slots.<br><br>
        <b>Text preprocessing:</b> Lowercase, remove punctuation, remove domain stopwords
        (words that appear universally in all reports with no discriminative value,
        e.g. "employee", "incident", "occurred").<br><br>
        <b>Models compared:</b> Logistic Regression and Random Forest, both using the
        same ColumnTransformer pipeline to prevent data leakage.<br><br>
        <b>Model selection criterion:</b> AUC-ROC on the unseen test set.
    </div>
    """, unsafe_allow_html=True)

    #Performance
    st.markdown("### Model Performance on Unseen Test Set")
    st.markdown("""
    <div class='info-box'>
    The test set (14,302 records) was held out completely during training and cross-validation.
    These results reflect how the model performs on data it has never seen before.
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Logistic Regression (Selected)**")
        st.markdown("""
        <table class='result-table'>
            <tr><th>Metric</th><th>Score</th></tr>
            <tr><td>Accuracy</td><td class='val-blue'>66.0%</td></tr>
            <tr><td>AUC-ROC</td><td class='val-blue'>0.7323</td></tr>
            <tr><td>Recall — High Priority</td><td class='val-blue'>68.8%</td></tr>
            <tr><td>Precision — High Priority</td><td class='val-blue'>65.2%</td></tr>
            <tr><td>F1 Score (weighted)</td><td class='val-blue'>66.0%</td></tr>
        </table>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("**Random Forest**")
        st.markdown("""
        <table class='result-table'>
            <tr><th>Metric</th><th>Score</th></tr>
            <tr><td>Accuracy</td><td class='val-blue'>63.7%</td></tr>
            <tr><td>AUC-ROC</td><td class='val-blue'>0.7101</td></tr>
            <tr><td>Recall — High Priority</td><td class='val-blue'>84.2%</td></tr>
            <tr><td>Precision — High Priority</td><td class='val-blue'>59.7%</td></tr>
            <tr><td>F1 Score (weighted)</td><td class='val-blue'>62.2%</td></tr>
        </table>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    #What the metrics mean
    st.markdown("### What Do These Metrics Mean?")
    st.markdown("""
    <div class='metric-card'>
        <strong>Accuracy</strong> — out of all 14,302 test records, what percentage did the model
        label correctly. LR correctly labelled 66% of all records.<br><br>
        <strong>AUC-ROC (0.7323)</strong> — measures how well the model separates High from Low Priority.
        A score of 0.5 means random guessing. A score of 1.0 means perfect separation.
        0.73 means the model correctly ranks a random High Priority incident above a random
        Low Priority incident 73% of the time.<br><br>
        <strong>Recall — High Priority (68.8%)</strong> — of all 7,151 actual High Priority incidents
        in the test set, the model correctly identified 4,921 of them (68.8%).
        It missed 2,230. In a safety context this is the most critical metric —
        a missed urgent incident is more dangerous than a false alarm.<br><br>
        <strong>Precision — High Priority (65.2%)</strong> — of all incidents the model flagged as
        High Priority, 65.2% were actually High Priority. The remaining 34.8% were
        false alarms that a safety officer would review and dismiss.<br><br>
        <strong>F1 Score</strong> — the balance between Precision and Recall. A high F1 means the
        model is both catching most urgent cases AND not raising too many false alarms.<br><br>
        <strong>Why Logistic Regression over Random Forest?</strong><br>
        LR was selected based on higher AUC-ROC (0.7323 vs 0.7101), meaning it
        discriminates better overall. However, Random Forest has a much higher Recall
        (84.2% vs 68.8%) — it catches far more urgent incidents at the cost of more
        false alarms. For a real safety deployment, RF's higher recall may be preferable
        since missing an urgent incident is more dangerous than reviewing a false alarm.
    </div>
    """, unsafe_allow_html=True)

    #Limitations
    st.markdown("### Limitations")
    st.markdown("""
    <div class='metric-card'>
        <strong>Vocabulary overlap:</strong> High and Low Priority incidents are described using
        very similar language. A strained knee and a fractured knee use almost identical
        words, making the classification task genuinely difficult.<br><br>
        <strong>Death cases:</strong> Only 29 Death cases appeared in the 100,000 record sample
        (0.03%). The model groups Deaths with DAFW under High Priority to mitigate this,
        but it may not reliably distinguish fatal incidents specifically.<br><br>
        <strong>Dataset scope:</strong> The model was trained on US OSHA data. Performance may
        differ on WestMine-specific incident reports which may use different terminology.<br><br>
        <strong>AI-assisted only:</strong> All classifications should be reviewed by a qualified
        safety officer before final determination.
    </div>
    """, unsafe_allow_html=True)



# PAGE 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Explorer":
    st.markdown("""
    <div class='page-title'>Dataset Explorer</div>
    <div class='page-subtitle'>Patterns in the OSHA ITA incident dataset used to train the model</div>
    <br>
    """, unsafe_allow_html=True)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Full Dataset",       "688,650 records")
    col2.metric("Training Sample",    "100,000 records")
    col3.metric("Balanced Dataset",   "~71,510 records")
    col4.metric("Test Set (unseen)",  "14,302 records")

    st.markdown("<br>", unsafe_allow_html=True)

    df = load_sample_data()

    #Outcome distribution 
    st.markdown("### Incident Outcome Distribution")
    fig1 = go.Figure(data=[go.Bar(
        x=['Days Away From Work (2)', 'Other Recordable (4)',
           'Job Transfer/Restriction (3)', 'Death (1)'],
        y=[35726, 34033, 30211, 29],
        marker_color=['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728'],
        text=[35726, 34033, 30211, 29],
        textposition='outside',
        textfont={'color': '#e8eaf0'}
    )])
    fig1.update_layout(**PLOTLY_LAYOUT,
        title='Incident Outcome Counts (100,000 record stratified sample)',
        yaxis_title='Number of Records')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    <div class='info-box'>
    Death cases (29 records, 0.03%) are almost invisible on a standard chart.
    This extreme imbalance is why Deaths and Days Away From Work are grouped together
    as High Priority — treating Death as its own class would give the model almost
    no examples to learn from.
    </div>
    """, unsafe_allow_html=True)

    #Binary label
    st.markdown("### Binary Priority Label Before Balancing")
    fig2 = go.Figure(data=[go.Bar(
        x=['High Priority (Death + DAFW)', 'Low Priority (Transfer + Other)'],
        y=[35755, 64244],
        marker_color=['#d62728', '#1f77b4'],
        text=['35,755 (35.8%)', '64,244 (64.2%)'],
        textposition='outside',
        textfont={'color': '#e8eaf0'}
    )])
    fig2.update_layout(**PLOTLY_LAYOUT,
        title='High vs Low Priority Before Downsampling',
        yaxis_title='Number of Records')
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    <div class='info-box'>
    Before balancing, Low Priority cases outnumber High Priority cases roughly 2:1.
    Downsampling the Low Priority class to match High Priority (35,755 each) ensures
    the model does not simply predict everything as Low Priority to achieve a high score.
    </div>
    """, unsafe_allow_html=True)

    #Incident type vs priority
    st.markdown("### Incident Type vs Priority Label")
    fig3 = go.Figure(data=[
        go.Bar(name='High Priority',
               x=['Respiratory', 'Poisoning', 'Other Illness', 'Injury', 'Skin Disorder', 'Hearing Loss'],
               y=[82.4, 27.3, 33.3, 35.2, 25.9, 9.5],
               marker_color='#d62728', opacity=0.85),
        go.Bar(name='Low Priority',
               x=['Respiratory', 'Poisoning', 'Other Illness', 'Injury', 'Skin Disorder', 'Hearing Loss'],
               y=[17.6, 72.7, 66.7, 64.8, 74.1, 90.5],
               marker_color='#1f77b4', opacity=0.85),
    ])
    fig3.update_layout(**PLOTLY_LAYOUT,
        barmode='stack',
        title='Incident Type vs Priority Label — % composition',
        yaxis_title='Percentage (%)',
        yaxis_range=[0, 100])
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("""
    <div class='info-box'>
    Respiratory conditions are 82.4% High Priority — the highest of any incident type.
    This makes sense: respiratory illnesses (including Covid) typically require days away from work.
    Hearing loss is 90.5% Low Priority — most hearing loss cases result in job transfer or
    restriction rather than full absence.
    </div>
    """, unsafe_allow_html=True)

    #Model performance
    st.markdown("### Model Performance Comparison")
    col5, col6 = st.columns(2)

    metrics = ['Accuracy', 'AUC-ROC', 'Recall\nHigh Priority',
               'Precision\nHigh Priority', 'F1 (weighted)']
    lr_scores = [0.6601, 0.7323, 0.6882, 0.6516, 0.6599]
    rf_scores = [0.6373, 0.7101, 0.8420, 0.5974, 0.6215]

    with col5:
        fig4 = go.Figure(go.Bar(
            x=lr_scores, y=metrics, orientation='h',
            marker_color='#1f77b4',
            text=[f"{v:.4f}" for v in lr_scores],
            textposition='outside', textfont={'color': '#e8eaf0'}
        ))
        layout4 = {**PLOTLY_LAYOUT}
        layout4['xaxis'] = {'range': [0, 1], 'gridcolor': '#2a2f3e', 'tickfont': {'color': '#8892a4'}}
        layout4['height'] = 320
        fig4.update_layout(**layout4, title='Logistic Regression')
        st.plotly_chart(fig4, use_container_width=True)

    with col6:
        fig5 = go.Figure(go.Bar(
            x=rf_scores, y=metrics, orientation='h',
            marker_color='#ff7f0e',
            text=[f"{v:.4f}" for v in rf_scores],
            textposition='outside', textfont={'color': '#e8eaf0'}
        ))
        layout5 = {**PLOTLY_LAYOUT}
        layout5['xaxis'] = {'range': [0, 1], 'gridcolor': '#2a2f3e', 'tickfont': {'color': '#8892a4'}}
        layout5['height'] = 320
        fig5.update_layout(**layout5, title='Random Forest')
        st.plotly_chart(fig5, use_container_width=True)

    if df is not None:
        st.markdown("### Live Data — Top States by Incident Count")
        if 'state' in df.columns:
            state_counts = df['state'].value_counts().head(15)
            fig6 = px.bar(x=state_counts.index, y=state_counts.values,
                          color=state_counts.values,
                          color_continuous_scale='Blues',
                          title='Top 15 States by Incident Count (dashboard sample)')
            fig6.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                               yaxis_title='Number of Records')
            st.plotly_chart(fig6, use_container_width=True)



# PAGE 3 — PREDICT INCIDENT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Incident":
    st.markdown("""
    <div class='page-title'>Incident Report Classifier</div>
    <div class='page-subtitle'>Enter the incident details below to classify priority level</div>
    <br>
    """, unsafe_allow_html=True)

    model = load_model()
    if model is None:
        st.error("Model file not found. Make sure models/best_model.joblib exists in the project folder.")
        st.stop()

    st.markdown("<div class='section-header'>Narrative Fields</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
        Fill in as many fields as possible. The more detail provided, the more accurate the prediction.
        The three fields marked with an asterisk carry the most weight in classification.
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        desc = st.text_area("Incident Description *",
            placeholder="e.g. Employee slipped on wet floor near loading dock",
            height=100)
        what_happened = st.text_area("What Happened *",
            placeholder="e.g. Worker was walking to the break room when they slipped on a wet floor and fell, striking their head on a metal shelf",
            height=120)
        illness = st.text_area("Injury / Illness Description *",
            placeholder="e.g. Laceration to forehead requiring 8 stitches, possible concussion",
            height=100)

    with col2:
        before = st.text_area("What Was Employee Doing Before?",
            placeholder="e.g. Walking from workstation to break room at end of shift",
            height=100)
        obj_substance = st.text_area("Object / Substance Involved",
            placeholder="e.g. Wet floor, metal shelf",
            height=100)
        location = st.text_area("Incident Location",
            placeholder="e.g. Warehouse floor near loading dock",
            height=80)

    st.markdown("<br><div class='section-header'>Severity Indicators</div>", unsafe_allow_html=True)
    col3, col4, col5 = st.columns(3)
    with col3:
        dafw = st.number_input("Days Away From Work", min_value=0, max_value=365, value=0)
    with col4:
        djtr = st.number_input("Days Job Transfer / Restriction", min_value=0, max_value=365, value=0)
    with col5:
        st.metric("Total Severity Days", dafw + djtr)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("CLASSIFY INCIDENT REPORT"):
        if not any([desc, what_happened, illness, before, obj_substance, location]):
            st.warning("Please fill in at least one narrative field before classifying.")
        else:
            # Build input — clip severity values same way as training
            p99_dafw_clip = 180
            p99_djtr_clip = 180

            input_df = pd.DataFrame([{
                "New_incident_description": clean_text(desc),
                "New_nar_before_incident": clean_text(before),
                "New_nar_what_happened": clean_text(what_happened),
                "New_nar_injury_illness": clean_text(illness),
                "New_nar_object_substance": clean_text(obj_substance),
                "New_incident_location": clean_text(location)
            }])
            
            prediction  = int(model.predict(input_df)[0])
            probability = model.predict_proba(input_df)[0]

            # Ensure probabilities are valid 0-1 values
            high_prob = float(np.clip(probability[1], 0.0, 1.0))
            low_prob  = float(np.clip(probability[0], 0.0, 1.0))

            st.markdown("<hr><div class='section-header'>Classification Result</div>",
                        unsafe_allow_html=True)

            if prediction == 1:
                st.markdown(f"""
                <div class='high-priority-card'>
                    <div class='priority-label' style='color:#ff4444;'>HIGH PRIORITY</div>
                    <div style='color:#ff8888; margin-top:8px; font-size:0.95rem;'>
                        This incident requires immediate attention and escalation.
                    </div>
                    <br>
                    <div style='color:#8892a4; font-size:0.78rem;
                                font-family:IBM Plex Mono,monospace; letter-spacing:2px;'>
                        CONFIDENCE
                    </div>
                    <div class='confidence-bar-container'>
                        <div class='confidence-bar-high' style='width:{high_prob*100:.1f}%'></div>
                    </div>
                    <div style='color:#ff4444; font-size:1.6rem;
                                font-family:IBM Plex Mono,monospace; font-weight:600;'>
                        {high_prob*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='low-priority-card'>
                    <div class='priority-label' style='color:#51cf66;'>LOW PRIORITY</div>
                    <div style='color:#88cc88; margin-top:8px; font-size:0.95rem;'>
                        This incident is recordable but does not require immediate escalation.
                    </div>
                    <br>
                    <div style='color:#8892a4; font-size:0.78rem;
                                font-family:IBM Plex Mono,monospace; letter-spacing:2px;'>
                        CONFIDENCE
                    </div>
                    <div class='confidence-bar-container'>
                        <div class='confidence-bar-low' style='width:{low_prob*100:.1f}%'></div>
                    </div>
                    <div style='color:#51cf66; font-size:1.6rem;
                                font-family:IBM Plex Mono,monospace; font-weight:600;'>
                        {low_prob*100:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

            #Probability breakdown table
            st.markdown("<br><div class='section-header'>Probability Breakdown</div>",
                        unsafe_allow_html=True)
            st.markdown(f"""
            <table class='result-table'>
                <tr>
                    <th>Class</th>
                    <th>Probability</th>
                    <th>Meaning</th>
                </tr>
                <tr>
                    <td><span class='val-high'>High Priority</span></td>
                    <td><span class='val-high'>{high_prob*100:.2f}%</span></td>
                    <td>Likelihood this is a Death or Days Away From Work case</td>
                </tr>
                <tr>
                    <td><span class='val-low'>Low Priority</span></td>
                    <td><span class='val-low'>{low_prob*100:.2f}%</span></td>
                    <td>Likelihood this is a Job Transfer or Other Recordable case</td>
                </tr>
                <tr>
                    <td>Predicted Label</td>
                    <td><span class='{'val-high' if prediction==1 else 'val-low'}'>
                        {'HIGH PRIORITY' if prediction==1 else 'LOW PRIORITY'}
                    </span></td>
                    <td>Class with the higher probability is selected</td>
                </tr>
            </table>
            <br>
            """, unsafe_allow_html=True)

            #What this means in plain language
            st.markdown("<div class='section-header'>What This Means</div>",
                        unsafe_allow_html=True)
            if prediction == 1:
                st.markdown(f"""
                <div class='metric-card'>
                The model predicts this incident is <b style='color:#ff4444'>High Priority</b>
                with {high_prob*100:.1f}% confidence.<br><br>
                High Priority incidents typically involve an employee requiring days away from
                work to recover, or in the most severe cases, a fatality. These cases require
                immediate review by a safety officer, formal documentation, and follow-up
                investigation to prevent recurrence.<br><br>
                The model is correct on High Priority cases approximately <b>68.8%</b> of the
                time on unseen data (Recall). A safety officer should always make the final call.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='metric-card'>
                The model predicts this incident is <b style='color:#51cf66'>Low Priority</b>
                with {low_prob*100:.1f}% confidence.<br><br>
                Low Priority incidents typically result in job transfer, restricted duty, or
                other recordable outcomes that do not require the employee to miss work entirely.
                These cases still require logging and follow-up but do not need immediate
                escalation.<br><br>
                The model is correct on Low Priority cases approximately <b>63.0%</b> of the
                time on unseen data. A safety officer should always make the final call.
                </div>
                """, unsafe_allow_html=True)
