# AERIS: Explainable Reasoning and Investigation System for Aerospace Maintenance

This repository contains the trained dual-stage machine learning inference engine for predictive aircraft maintenance using NGAFID telemetry.

---

## Quickstart Guide (For You and Your Colleague)

### 1. Prerequisites & Environment Setup
Clone or copy this folder to your machine, open it in **VS Code**, and open a terminal:

```bash
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate      # On Windows
# source venv/bin/activate  # On macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

### 2. Add Downloaded Artifacts
Place the downloaded artifacts from Kaggle (`aeris_project_artifacts.zip` extracted) into this folder:
* `aeris_stage1_anomaly_lgbm.pkl` (Stage 1 Anomaly Detector)
* `aeris_stage2_subsystem_lgbm.pkl` (Stage 2 Subsystem Classifier)
* `aeris_ngafid_final_predictions.csv` (Validation predictions)
* `aeris_fault_evidence_object.json` (Structured evidence object)

### 3. Run Inference
In your VS Code terminal, run:

```bash
python predict.py
```

This will load the dual-stage models and print the full flight inspection report with confidence bars and Top-3 differential subsystem candidates.

---

## Architecture Summary
* **Stage 1 (Anomaly Screening)**: Evaluates pre-failure operational risk (**75.3% accuracy, 0.7019 ROC-AUC**).
* **Stage 2 (Fault Isolation)**: Ranks failing mechanical subsystems (**79.01% Top-3 diagnostic coverage**).
* **SHAP Explainability Layer**: Generates exact Shapley attributions for evidence fusion.
