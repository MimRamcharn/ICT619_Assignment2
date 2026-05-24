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
Safe Report AI is a Streamlit web application that classifies workplace incident reports as either **High Priority** or **Low Priority**. The system uses a machine learning pipeline trained on OSHA ITA Case Detail Data 2024–2025.

The app is designed to support initial incident triage by analysing OSHA-style incident narratives and predicting whether the report may require immediate escalation.

## Project Structure

```text
ICT619_Assignment2/
│
├── .devcontainer/
├── SafeReportAI/
│   ├── app.py
│   ├── requirements.txt
│   ├── Assignment_2_final.ipynb
│   └── models/
│       └── best_model.joblib
│
└── README.md
```

The minimum files needed to run the Streamlit app are:

```text
app.py
requirements.txt
models/best_model.joblib
```

---

## Model Details

- **Model type:** Logistic Regression with TF-IDF features
- **Pipeline:** Scikit-learn Pipeline with ColumnTransformer
- **Task:** Binary text classification
- **Input:** Six OSHA-style narrative text fields
- **Output:** High Priority (1) or Low Priority (0)
- **AUC-ROC:** 0.7323 on the unseen test set
- **Dataset:** OSHA ITA Case Detail Data 2024–2025

### Model Input Fields

The deployed model expects the following six columns:

```text
New_incident_description
New_nar_before_incident
New_nar_what_happened
New_nar_injury_illness
New_nar_object_substance
New_incident_location
```
---

## Target Variable

| Label | Outcome Codes | Meaning |
|---|---|---|
| High Priority (1) | 1, 2 | Death or Days Away From Work |
| Low Priority (0) | 3, 4 | Job Transfer/Restriction or Other Recordable Case |

---

## How to Run the Project Locally

These steps can be used to run the app on your own computer.

### 1. Download or clone the GitHub repository

```bash
git clone https://github.com/MimRamcharn/ICT619_Assignment2.git
cd ICT619_Assignment2 then move to SafeReportAI
```

Alternatively, download the repository as a ZIP file and open the project folder.

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Make sure the model file exists

The app needs the trained model file here:

```text
models/best_model.joblib
```

There are two ways to get this file:

**Option A — Use the existing model file**

The `models` folder already contains `best_model.joblib`, no notebook rerun is needed.

**Option B — Regenerate the model from the notebook**

Run:

```text
notebooks/Assignment_2_final.ipynb
```

This notebook trains the model and saves the final pipeline as:

```text
models/best_model.joblib
```

### 4. Start the Streamlit app

```bash
streamlit run app.py
```

After running this command, Streamlit will open the app in your browser.

---

## How to Deploy on Streamlit Cloud

The app can be deployed online using GitHub and Streamlit Cloud.

### Option 1 — Deploy using the existing app files

As the model has already been generated, you do not need to rerun the notebook. Upload the following files and folders to GitHub:

```text
app.py
requirements.txt
models/best_model.joblib
```

Then connect the GitHub repository to Streamlit Cloud.

### Option 2 — Regenerate the model before deployment

Run the notebook first:

```text
notebooks/Assignment_2_final.ipynb
```

This will create the trained model file:

```text
models/best_model.joblib
```

After that, upload the `models` folder, `requirements.txt`, and `app.py` to GitHub.

### Streamlit Cloud deployment steps

1. Push the project folder to a GitHub repository.
2. Go to Streamlit Cloud.
3. Connect your GitHub account.
4. Select the repository.
5. Set the main file path as:

```text
SafeReportAI/app.py
```

6. Click **Deploy**.
7. Once deployed, the app will open on the Streamlit platform and can be accessed through the generated website link.

---

## Accessing the Online App

The app can be accessed through the deployed Streamlit website link:

```text
https://github.com/MimRamcharn/ICT619_Assignment2.git
```

No need to run the notebook or install anything locally if using the deployed website version.

---

## Important Notes

- The app is designed for OSHA-style workplace incident reports.
- The model always predicts either High Priority or Low Priority, even if unrelated text is entered.
