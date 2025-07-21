# Advanced Fraud Detection System for E-commerce & Banking

## Overview

This project delivers a comprehensive fraud detection solution for Adey Innovations Inc., designed to protect against financial losses in e-commerce and banking. By integrating advanced data analysis, sophisticated feature engineering, and powerful machine learning models, this system accurately identifies fraudulent transactions while ensuring a seamless experience for legitimate users. The core of this solution lies in its ability to not only detect fraud but also explain *why* a transaction is flagged, providing actionable insights through model explainability with SHAP.

---

## Project Structure

The repository is organized to ensure clarity and reproducibility:


---

## Methodology and Workflow

The project follows a structured, three-phase workflow from data ingestion to actionable insights.

#### **Phase 1: Data Analysis and Preprocessing**
*   **Data Integrity:** Performed rigorous data cleaning, handled missing values, and removed duplicates to establish a reliable foundation.
*   **Exploratory Data Analysis (EDA):** Conducted in-depth EDA with visualizations to uncover patterns, correlations, and anomalies in the transaction data.
*   **Feature Engineering:** Created high-impact features, including transaction frequency, time-based patterns, and user geolocation by mapping IP addresses.
*   **Data Transformation:** Normalized numerical features and applied SMOTE (Synthetic Minority Over-sampling Technique) to address severe class imbalance, preventing model bias towards non-fraudulent transactions.

#### **Phase 2: Model Building and Training**
*   **Model Selection:** Implemented two distinct models to benchmark performance:
    *   **Logistic Regression:** A robust linear model serving as a performance baseline.
    *   **Random Forest:** A powerful ensemble model chosen for its high accuracy and ability to capture complex, non-linear relationships.
*   **Model Evaluation:** Assessed model performance using metrics crucial for imbalanced datasets: **F1-Score**, **Area Under the Precision-Recall Curve (AUC-PR)**, and the **Confusion Matrix**.
*   **Comparative Analysis:** The Random Forest model demonstrated superior performance over Logistic Regression across all key metrics.

#### **Phase 3: Model Explainability and Interpretation**
*   **Unlocking the "Black Box":** Utilized **SHAP (SHapley Additive exPlanations)** to interpret the Random Forest model’s predictions.
*   **Actionable Insights:** Generated summary, beeswarm, and force plots to understand both global feature importance and local, per-transaction fraud drivers.
*   **Bridging Technical and Business Needs:** Translated complex SHAP findings into clear business recommendations, empowering fraud analysts and stakeholders.

---

## Key Results

*   **Superior Model Performance:** The Random Forest classifier significantly outperformed the Logistic Regression baseline, achieving a higher F1-score and AUC-PR, indicating its effectiveness in accurately identifying fraud while minimizing false alarms.
*   **Actionable Fraud Drivers:** SHAP analysis successfully identified the key features influencing fraudulent predictions. These insights provide a clear, data-driven basis for refining fraud prevention rules and supporting operational decisions.

---

## How to Use This Project

1.  **Setup:** Clone the repository and place the required datasets (`Fraud_Data.csv`, `IpAddress_to_Country.csv`, `creditcard.csv`) into the `data/` directory.
2.  **Execution:** Run the Jupyter notebooks in the `notebooks/` folder sequentially to replicate the analysis, model training, and interpretation.
3.  **Review:** Consult the professional reports in the `reports/` folder for high-level summaries and detailed findings.

---

## Business Value for Adey Innovations Inc.

This project provides a multi-faceted solution that drives significant business value:

*   **Enhanced Fraud Detection:** Proactively identifies and mitigates fraudulent activity with superior accuracy, directly reducing financial losses.
*   **Improved Customer Trust:** Minimizes false positives, ensuring legitimate customer transactions are not unnecessarily blocked, thereby enhancing user experience and building brand loyalty.
*   **Operational Efficiency:** Equips fraud investigation teams with clear, data-driven insights, allowing them to prioritize and resolve cases more effectively.
*   **Strategic Decision-Making:** Delivers a deep understanding of fraud patterns, enabling Adey Innovations Inc. to develop smarter, more adaptive security protocols.