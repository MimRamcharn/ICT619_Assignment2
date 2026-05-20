# ICT619_Assignment2
# Safe Report AI — WestMine Incident Classifier

## NLP-based workplace incident classification system built for ICT619 Assignment 2.

Date: 22 May 2026

Members:

| Name | Student ID |
|----------|----------|
| Mimansha D. Ramcharn | 35437257 |
| Sarwar Taki    | 35405739 |
| Maruf HASAN    | 35397276 |

## Project Structure
## How to Run Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the notebook first to generate the model:
```
notebooks/Assignment_2_final.ipynb
```
To save `models/best_model.joblib`

3. Launch the app:
```bash
streamlit run app.py
```

## How to Deploy on Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to share.streamlit.io
3. Connect to GitHub account
4. Select the repository and set main file to `app.py`
5. Click Deploy


## Model Details

- **Type:** Logistic Regression + TF-IDF (ColumnTransformer)
- **Input:** 6 narrative text fields + 3 numeric severity features
- **Output:** High Priority (1) or Low Priority (0)
- **AUC-ROC:** 0.7323 on unseen test set
- **Dataset:** OSHA ITA Case Detail Data 2024-2025

## Target Variable

| Label | Outcome Codes | Meaning |
|-------|--------------|---------|
| High Priority (1) | 1, 2 | Death or Days Away From Work |
| Low Priority (0) | 3, 4 | Job Transfer/Restriction or Other Recordable |
