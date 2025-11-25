# 📈 Airbnb Listing Success Predictor: Applied Data Science MSc Challenge (London Data)


## 🌟 Project Overview

This repository contains the complete workflow for my **Applied Data Science MSc Coursework**, focusing on predicting the success of Airbnb listings in London. The project follows the full data science lifecycle, from **advanced feature engineering** to **model interpretation** using specific business-driven metrics.

**Goal:** Build a robust classification model to estimate a listing's probability of being a **'Good Listing'** (defined as a top 25% performer).

**Key Technologies:** R (dplyr, caret, rpart, pROC), Advanced XGBoost Imputation.

---

## 🎯 Methodology: Defining Success

The project's foundation is a highly defensible **Composite Success Score ($\mathbf{S_i}$)** that balances business goals and customer satisfaction.

$$\mathbf{S_i} = 0.5 \times \text{Normalized}(\text{Demand and Value Score}) + 0.5 \times \text{Normalized}(\text{Quality and Host Score})$$

| Component | Perspective | Original Column(s) Used | Custom Derived Feature |
| :--- | :--- | :--- | :--- |
| **Demand & Value ($\mathbf{DS_i}$)** | **Business & Host** | $\text{availability 365, price, accommodates}$ | $\mathbf{\text{Booking Rate Proxy}}$ and $\mathbf{\text{Value Density}}$ ($\text{price} / \text{accommodates}$). |
| **Quality & Host ($\mathbf{QS_i}$)** | **Customer** | $\text{review scores rating, host acceptance rate, host is superhost}$ | $\mathbf{\text{Avg Review Score}}$ ($\text{3 scores}$) and $\mathbf{\text{Host Status Premium}}$. |

The final classification of **'Good'** was set at the **75th Percentile** of this composite score.

---

## 🛠️ Data Preparation and Engineering Excellence

Data cleaning and engineering were the most challenging and innovative parts of this project, focusing on quality and maximizing data retention.

### 1. Advanced Missing Data Handling

* **Initial Cleanup:** I manually dropped $\mathbf{30}$ noisy/irrelevant columns (e.g., all $\text{URLs, descriptions, messy location data}$ and $\text{availability 30/60/90}$) to focus the model.
* **XGBoost Imputation:** Instead of losing over $\mathbf{17,000}$ rows with missing $\text{bedrooms}$, I used a supervised **XGBoost Regressor** to accurately predict these $\text{NA}$ values based on property size proxies ($\text{price, beds, bathrooms}$). This advanced technique preserved $\mathbf{\sim 35\%}$ of my original data.
    * *Resulting Imputation Accuracy:* $\mathbf{R² = 0.904}$

### 2. Feature Creation & Imbalance Solution

* **Final Predictors:** I engineered crucial features like $\mathbf{\text{host tenure days}}$, $\mathbf{\text{review recency days}}$, $\mathbf{\text{verification count}}$, and $\mathbf{10\ binary\ amenity\ flags}$ (e.g., $\text{has dryer, has wifi}$).
* **Target Balancing:** The final dataset had a $\mathbf{3:1}$ imbalance ($\text{Bad}$ vs. $\text{Good}$). I implemented **Undersampling** to create a balanced $\mathbf{1:1}$ training set, ensuring my models focused equally on correctly identifying the high-value 'Good' listings.

---

## 📊 Modeling and Interpretation

The models were strategically chosen to meet both prediction and interpretation goals. Both were trained on the $\mathbf{1:1}$ balanced dataset.

## 4. 📈 Final Model Performance Comparison

The models were trained on the balanced $\mathbf{1:1}$ dataset to ensure they could accurately predict the 'Good' (minority) class. The evaluation confirms the superior generalized performance of the Decision Tree for this specific classification task.

| Model | Final Role | Balanced Accuracy | AUC-ROC | Specificity ('Good' Prediction Rate) |
| :--- | :--- | :--- | :--- | :--- |
| **Optimized Decision Tree** ($\text{rpart}$) | **Final Predictive Model** | $\mathbf{73.48\%}$ | $\mathbf{0.7747}$ | $\mathbf{75.74\%}$ |
| **Optimized Logistic Regression** ($\text{glm}$) | **Interpretive Model** | $71.04\%$ | $0.7743$ | $78.54\%$ |

### Model Selection Rationale

The **Optimized Decision Tree** is selected as the primary predictive model for deployment because it achieved the highest **Balanced Accuracy** ($73.48\%$) and **AUC-ROC** ($0.7747$). This means it is the most effective at identifying both low and high-performing listings equally.

The **Optimized Logistic Regression** is kept as the essential **interpretive model**. Its coefficients provide the clear, direct $\mathbf{\beta}$ values necessary for generating specific, actionable advice to hosts (e.g., "Increase the log-odds of success by $27\%$ by adding a dryer").

### 2. Actionable Insights

The model coefficients revealed the exact levers for host success:

1.  **Demand Dominates:** $\mathbf{\text{booking rate proxy}}$ is the single most powerful positive predictor.
2.  **Service is Priceless:** Offering $\mathbf{\text{Dryer}}$ and having $\mathbf{\text{Air\ Conditioning}}$ (high $\beta$ coefficients) significantly increases success likelihood.
3.  **Consistency Matters:** High $\mathbf{\text{review recency days}}$ (time since last review) is the biggest negative predictor, confirming that continuous host engagement is essential.

### 3. Business Rule (Decision Tree)

The Decision Tree provides a simple path to success: $\mathbf{\text{High Occupancy}}$ $\rightarrow$ $\mathbf{\text{Proven Track Record}}$ $\rightarrow$ $\mathbf{\text{Success}}$.

---

## 🚀 Productization and Next Steps 

This project is ready for deployment as an automated tool called **The Airbnb Doctor**.

**Deployment Plan:**
1.  **Model:** Deploy the $\mathbf{\text{Optimized Decision Tree}}$ model.
2.  **Functionality:** A host provides their listing details, and the tool returns a $\mathbf{\text{Predicted Success Score}}$.
3.  **Value:** The system uses the $\mathbf{\text{GLM coefficients}}$ to generate specific, actionable advice (e.g., "Your success probability is $\mathbf{60\%}$. To reach $75\%$, our analysis suggests $\mathbf{\text{getting your Host Identity Verified}}$ and $\mathbf{\text{adding an Essentials amenity}}$").
