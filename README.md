# 📈 Airbnb Listing Success Predictor
### Applied Data Science | MSc Coursework (London Dataset)

<p align="center">
<img src="https://img.shields.io/badge/R-Language-276DC3?style=for-the-badge&logo=r">
<img src="https://img.shields.io/badge/XGBoost-Imputation-111111?style=for-the-badge">
<img src="https://img.shields.io/badge/Model-Decision%20Tree%20%26%20GLM-CC0000?style=for-the-badge">
<img src="https://img.shields.io/badge/Accuracy-73.48%25-green?style=for-the-badge">
<img src="https://img.shields.io/badge/Data-London%20Airbnb-FF5A5F?style=for-the-badge&logo=airbnb">
</p>

---

## 🌟 Project Overview

This repository features a complete end-to-end data science workflow designed to predict the success of Airbnb listings in London. By applying advanced feature engineering and machine learning interpretation, the project identifies the "Top 25%" high-performing listings (defined as **'Good Listings'**).

**The Challenge:** Navigate high missing-data rates and complex feature interactions to provide actionable business intelligence for property hosts.

---

## 🎯 Methodology: Defining Success

We established a **Composite Success Score ($S_i$)** to balance commercial performance with guest satisfaction:

$$S_i = 0.5 \times \text{Normalized}(\text{Demand/Value}) + 0.5 \times \text{Normalized}(\text{Quality/Host})$$



### Score Components
| Component | Focus | Custom Derived Features |
| :--- | :--- | :--- |
| **Demand & Value ($DS_i$)** | **Commercial** | Booking Rate Proxy, Value Density (Price/Accomm) |
| **Quality & Host ($QS_i$)** | **Experience** | Avg Review Score (3-score aggregate), Host Premium Status |

---

## 🛠️ Data Engineering Excellence

### 1️⃣ Advanced Missing Data Handling
* **High-Retention Imputation:** Instead of discarding **17,000+** rows with missing `bedrooms`, I utilized an **XGBoost Regressor** ($R^2 = 0.904$) to predict counts based on price, beds, and bathrooms.
* **Impact:** Preserved **~35%** of the original dataset that would have otherwise been lost.

### 2️⃣ Feature Creation & Balancing
* **Engineered Predictors:** Created `host_tenure_days`, `review_recency`, `verification_count`, and **10 binary amenity flags**.
* **Target Balancing:** Addressed a **3:1 imbalance** via **Undersampling**, creating a robust 1:1 balanced training set for model stability.

---

## 📊 Modeling and Interpretation

The strategy utilized two distinct models: one for **Maximum Prediction** and one for **Human Interpretation**.

### 📈 Model Performance Comparison

| Model | Role | Balanced Accuracy | AUC-ROC | Specificity |
| :--- | :--- | :--- | :--- | :--- |
| **Optimized Decision Tree** | **Primary Predictor** | **73.48%** | **0.7747** | **75.74%** |
| **Logistic Regression (GLM)** | **Interpretive Engine** | 71.04% | 0.7743 | 78.54% |

---

## 🔍 Actionable Insights

Based on the **GLM Coefficients ($\beta$)**, the following levers directly influence listing success:

1. **Demand Dominance:** The `booking_rate_proxy` is the single strongest indicator of a "Good" listing.
2. **The "Amenity Premium":** Adding a **Dryer** or **Air Conditioning** significantly shifts the log-odds of success.
3. **Engagement Gap:** High `review_recency` (long gaps between reviews) is the most aggressive negative predictor.

---

## 🚀 Productization: "The Airbnb Doctor"

This project is architected for deployment as a diagnostic tool for hosts:

* **Input:** Host provides listing metadata (amenities, price, frequency).
* **Process:** The **Decision Tree** classifies the listing; the **GLM** identifies specific weaknesses.
* **Output:** A personalized "Success Roadmap" (e.g., *"Your success probability is 60%. Add 'Essentials' and verify your identity to reach 75%"*).

---

## 👨‍💻 Author

**Narendra Gandikota (G‑Narendra)** Applied Data Science | ML | Python | R | GenAI Specialist  

GitHub: [https://github.com/G-Narendra](https://github.com/G-Narendra)
