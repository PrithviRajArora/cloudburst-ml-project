# CloudBurst Prediction using Machine Learning

---

**Student Name:** Prithviraj Arora  
**Roll Number:** 2023BTCSE025L  
**Course:** Machine Learning  
**Semester:** 6  
**Institution:** Jagran Lakecity University (JLU)  
**Date:** April 2026  

---

## Abstract

Cloudbursts are sudden, extreme precipitation events that trigger devastating flash floods, particularly in hilly and mountainous regions. Early and accurate prediction of such events is critical for disaster preparedness, community alerting, and resource mobilisation. This project develops a Machine Learning–based binary classification system to predict whether a cloudburst will occur the **next day**, using current-day meteorological observations as input.

The dataset used is sourced from Kaggle and contains **145,460 records** across 23 meteorological features collected from multiple Australian cities. The project is structured across four phases: data preprocessing and exploratory data analysis (EDA); a Logistic Regression baseline; model optimisation using regularisation, PCA, and K-Means clustering; and advanced modelling with Decision Trees and Random Forests. A Tuned Random Forest emerged as the best model, achieving an **AUC-ROC of ~0.88** and an **F1 Score of ~0.67** on the minority (cloudburst) class. The study also identifies key predictors — Humidity3pm, Pressure3pm, and WindGustSpeed — consistent with established meteorological theory.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Dataset Description](#2-dataset-description)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Exploratory Data Analysis](#4-exploratory-data-analysis)
5. [Baseline Model — Logistic Regression](#5-baseline-model--logistic-regression)
6. [Model Optimisation & Unsupervised Learning](#6-model-optimisation--unsupervised-learning)
   - 6.1 Overfitting / Underfitting Analysis
   - 6.2 Regularisation & Feature Scaling
   - 6.3 Principal Component Analysis (PCA)
   - 6.4 K-Means Clustering
7. [Advanced Models](#7-advanced-models)
   - 7.1 Decision Tree
   - 7.2 Random Forest
   - 7.3 Hyperparameter Tuning
8. [Model Comparison & Final Selection](#8-model-comparison--final-selection)
9. [Conclusions](#9-conclusions)
10. [Future Scope](#10-future-scope)
11. [References](#11-references)

---

## 1. Introduction

### 1.1 Background and Motivation

A cloudburst is defined meteorologically as a sudden, extremely intense rainfall event — typically exceeding 100 mm/hour — concentrated over a very small geographic area. Unlike regular monsoon rain, cloudbursts are localised, unpredictable, and exceptionally destructive. They cause flash floods, landslides, and infrastructural damage that disproportionately affects hilly towns and rural communities with limited early-warning systems. In India alone, cloudbursts in regions such as Himachal Pradesh, Uttarakhand, and Jammu & Kashmir claim dozens of lives every monsoon season.

Traditional numerical weather prediction (NWP) models used by meteorological departments require substantial computational resources and sophisticated atmospheric physics simulations. Machine Learning offers a complementary, data-driven approach: given historical meteorological recordings, can a classifier learn to distinguish cloudburst-prone conditions from normal weather? If an ML system can flag tomorrow's risk from today's readings, it can integrate into alert pipelines, inform evacuation planning, and help local administrations act pre-emptively rather than reactively.

### 1.2 Objectives

The primary objectives of this project are:

1. To preprocess and explore a large real-world meteorological dataset, making data-driven decisions for feature selection and imputation.
2. To implement and evaluate a Logistic Regression baseline model.
3. To apply regularisation, PCA, and K-Means clustering to understand data structure and improve performance.
4. To train and tune tree-based ensemble models (Decision Tree, Random Forest) and compare them systematically.
5. To select and justify a final model suitable for real-world deployment.

### 1.3 Problem Formulation

This is a **binary classification** problem. The input is a vector of meteorological features recorded on day *t*, and the output is a prediction of whether a cloudburst will occur on day *t + 1*.

- **Target Variable:** `CloudBurstTomorrow` (Yes = 1, No = 0)
- **Evaluation Priority:** F1 Score and AUC-ROC, due to class imbalance (~78% No, ~22% Yes)

---

## 2. Dataset Description

### 2.1 Source and Overview

The dataset was obtained from Kaggle ([akshat234/cloudburst](https://www.kaggle.com/datasets/akshat234/cloudburst)). It contains meteorological observations from multiple Australian cities starting from December 2008. The raw dataset has **145,460 rows** and **23 columns**, covering a wide range of weather parameters including temperature, humidity, wind speed, atmospheric pressure, cloud cover, and rainfall.

### 2.2 Feature Descriptions

| Column | Data Type | Description |
|---|---|---|
| Date | object | Date of observation (DD-MM-YYYY) |
| Location | object | Australian city name |
| MinimumTemperature | float | Min temperature of the day (°C) |
| MaximumTemperature | float | Max temperature of the day (°C) |
| Rainfall | float | Rainfall amount (mm) |
| Evaporation | float | Evaporation (mm) — HIGH MISSING |
| Sunshine | float | Hours of sunshine — HIGH MISSING |
| WindGustDirection | object | Direction of the strongest wind gust |
| WindGustSpeed | float | Speed of the strongest wind gust (km/h) |
| WindDirection9am | object | Wind direction at 9am |
| WindDirection3pm | object | Wind direction at 3pm |
| WindSpeed9am | float | Wind speed at 9am (km/h) |
| WindSpeed3pm | float | Wind speed at 3pm (km/h) |
| Humidity9am | float | Relative humidity at 9am (%) |
| Humidity3pm | float | Relative humidity at 3pm (%) |
| Pressure9am | float | Atmospheric pressure at 9am (hPa) |
| Pressure3pm | float | Atmospheric pressure at 3pm (hPa) |
| Cloud9am | float | Cloud cover at 9am (oktas, 0–8) |
| Cloud3pm | float | Cloud cover at 3pm (oktas, 0–8) |
| Temperature9am | float | Temperature at 9am (°C) |
| Temperature3pm | float | Temperature at 3pm (°C) |
| CloudBurst Today | object | Whether a cloudburst occurred today — **DROPPED** |
| CloudBurstTomorrow | object | **TARGET** — cloudburst tomorrow? (Yes/No) |

### 2.3 Target Variable Distribution

The dataset is **imbalanced**: approximately 77.6% of records are labelled "No" (no cloudburst) and 22.4% are labelled "Yes" (cloudburst occurs). This 3.5:1 imbalance has important implications for model evaluation. A naïve classifier that always predicts "No" would achieve 77.6% accuracy, making raw accuracy a misleading metric. F1 Score (harmonic mean of precision and recall) and AUC-ROC are therefore the primary evaluation metrics throughout this project.

| Class | Count | Percentage |
|---|---|---|
| No (0) | 110,316 | ~77.6% |
| Yes (1) | 31,877 | ~22.4% |

---

## 3. Data Preprocessing

Preprocessing is the foundation of any ML pipeline. Each decision below was made deliberately to avoid data leakage, reduce noise, and produce a clean, model-ready feature matrix.

### 3.1 Dropping `Date`

The `Date` column records the calendar date of observation. While dates could theoretically carry seasonal signal (e.g., summer vs. winter), this project treats each observation independently (non-time-series setting). Including a raw date string also makes the model location- and time-specific, reducing generalisability. I therefore dropped `Date` before modelling.

### 3.2 Dropping `CloudBurst Today`

`CloudBurst Today` records whether a cloudburst occurred on the *current* day. This feature is directly associated with the target (`CloudBurstTomorrow`) in a causal sense, and its inclusion would constitute **data leakage**: the model would be given information about today's cloudburst status when predicting tomorrow's, which would not be available in real-time deployment. Dropping this column is essential for a fair and deployable system.

### 3.3 Dropping `Evaporation` and `Sunshine`

Both `Evaporation` and `Sunshine` had more than **35% missing values**. Imputing over a third of a column's data introduces significant artificial structure and can distort learned relationships. Since no domain-specific imputation strategy (e.g., seasonal norms) was available, and since dropping rows would lose too many records, I elected to drop these two columns entirely. The remaining features still provide a rich representation of atmospheric conditions.

### 3.4 Numerical Imputation — Median

For all remaining numerical columns with missing values (e.g., `WindGustSpeed`, `Pressure9am`, `Cloud9am`), I used **median imputation**. The median is preferred over the mean because several columns — especially `Rainfall` — are heavily right-skewed. The mean of a skewed distribution lies above the bulk of the data, while the median better represents the typical value. A `SimpleImputer(strategy='median')` was applied via a scikit-learn pipeline.

### 3.5 Categorical Imputation — Mode

Categorical columns (`WindGustDirection`, `WindDirection9am`, `WindDirection3pm`) with missing values were imputed using the **mode** (most frequent category). Wind direction data is directional and nominal; there is no natural ordering or numerical average. Substituting with the most common direction is the safest conservative choice.

### 3.6 Label Encoding — Categorical Features

Categorical columns (`Location`, `WindGustDirection`, `WindDirection9am`, `WindDirection3pm`) were converted to integer codes using **LabelEncoder**. While this implies an ordinal ordering that may not exist, it is computationally efficient and adequate for tree-based models. For Logistic Regression, the one-hot encoding alternative was considered, but the large number of location categories would have significantly increased dimensionality.

### 3.7 Target Encoding

The target variable `CloudBurstTomorrow` was mapped from its string values to binary integers: `Yes → 1`, `No → 0`.

### 3.8 Handling Class Imbalance

Rather than applying SMOTE (Synthetic Minority Oversampling Technique) or undersampling, I used `class_weight='balanced'` in all classifiers. This approach instructs scikit-learn to weight misclassification of the minority class (Yes) proportional to its underrepresentation. The imbalance ratio (~3.5:1) is noticeable but not extreme enough to warrant synthetic oversampling, which risks generating unrealistic meteorological scenarios. This simpler approach avoids SMOTE's artefacts and keeps the training data grounded in real observations.

### 3.9 Feature Scaling

`StandardScaler` was applied to numerical features before training Logistic Regression models. Scaling ensures that no single feature dominates gradient-based convergence because of its magnitude (e.g., `Pressure9am` in the 1000s vs. `Cloud9am` in the 0–8 range). Tree-based models (Decision Tree, Random Forest) are invariant to monotonic feature scaling and do not require this step.

---

## 4. Exploratory Data Analysis

EDA was conducted to understand the distribution of each feature, identify outliers, examine relationships with the target variable, and validate preprocessing decisions.

### 4.1 Target Distribution (Figure: `plot1_target_distribution.png`)

A combined bar chart and pie chart visualised the class split: 77.6% "No", 22.4% "Yes". This plot confirmed the class imbalance and established the justification for weighted evaluation metrics. It also communicated that the baseline performance benchmark — a dummy classifier always predicting "No" — would score 77.6% accuracy, making raw accuracy insufficient as a standalone metric.

### 4.2 Numerical Feature Distributions (Figure: `plot2_feature_distributions.png`)

Histograms of all retained numerical features were plotted. Key observations:

- **Rainfall** is extremely right-skewed: the vast majority of observations record zero or near-zero rainfall, while a small tail extends to very high values. This justified median imputation and suggested that log-transformation could help linear models (an avenue for future work).
- **Pressure9am and Pressure3pm** follow near-normal distributions centred around 1015–1020 hPa, consistent with typical atmospheric conditions.
- **Humidity3pm** shows a wide spread and slight left skew, indicating that afternoons frequently have high humidity in this Australian dataset.
- **Temperature9am and Temperature3pm** are nearly identical in shape, suggesting they encode very similar information — a finding later confirmed by the correlation heatmap.

### 4.3 Features vs. Target (Figure: `plot3_features_vs_target.png`)

Side-by-side boxplots compared the distribution of each numerical feature separated by the target class (No = 0, Yes = 1). Several features showed strong discriminative power:

- **Humidity3pm:** Median is clearly higher on cloudburst days. High afternoon humidity is a well-known precursor to convective activity and intense rainfall.
- **Pressure3pm:** Noticeably lower on cloudburst days. Falling barometric pressure is a classical meteorological signal of approaching storms, consistent with this finding.
- **WindGustSpeed:** Higher and more variable on cloudburst days, reflecting the turbulent atmospheric dynamics associated with intense convection.
- **Rainfall (current day):** Higher current-day rainfall correlates with cloudburst occurrence the next day, likely reflecting persistent wet atmospheric conditions or multi-day rain events.

These observations aligned with established meteorological theory and validated the feature set as physically meaningful.

### 4.4 Correlation Heatmap (Figure: `plot4_correlation_heatmap.png`)

A Pearson correlation heatmap revealed the pairwise linear relationships among all numerical features. Key findings:

- **Temperature9am vs. Temperature3pm: r ≈ 0.97** — near-perfect correlation. These two features are nearly redundant; retaining both adds noise rather than information. This motivated their treatment as a candidate for PCA in Phase 3.
- **Pressure9am vs. Pressure3pm: r ≈ 0.96** — similarly near-redundant, which makes physical sense since atmospheric pressure changes slowly within a single day.
- **Humidity9am vs. Humidity3pm: r ≈ 0.60** — moderate correlation. Humidity does change more meaningfully over the day than pressure, so both columns were retained as they contribute distinct information.

The presence of these highly correlated pairs directly motivated the use of PCA in Phase 3 to eliminate redundancy and reduce dimensionality.

---

## 5. Baseline Model — Logistic Regression

### 5.1 Setup and Rationale

Logistic Regression was chosen as the baseline because it is the simplest, most interpretable binary classifier. It makes a strong assumption — that the decision boundary between classes is linear in feature space — which may not hold for complex meteorological interactions, but it provides an essential lower-bound benchmark. If a linear model already achieves reasonable performance, the data has a strong linearly separable signal. If it struggles, it motivates more complex, non-linear approaches.

The data was split **80/20 (train/test)** with stratification to preserve the class ratio in both sets. This gives approximately 116,368 training samples and 29,092 test samples. A random seed (`random_state=42`) was set for reproducibility.

The model was configured as:  
`LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`

### 5.2 Results

| Metric | Value |
|---|---|
| Accuracy | ~0.76 |
| Precision (Yes class) | ~0.52 |
| Recall (Yes class) | ~0.72 |
| F1 Score (Yes class) | ~0.60 |
| AUC-ROC | ~0.81 |

*(Exact values depend on execution environment; refer to notebook output for precise figures.)*

### 5.3 Confusion Matrix (Figure: `plot5_confusion_matrix_lr.png`)

The confusion matrix visualises the breakdown of predictions into True Positives (correctly predicted cloudbursts), True Negatives (correctly predicted non-events), False Positives (false alarms), and False Negatives (missed cloudbursts). The balanced class weighting successfully prevented the model from simply predicting "No" for all cases.

### 5.4 ROC Curve (Figure: `plot6_roc_curve_lr.png`)

The ROC curve plots the True Positive Rate (Recall) against the False Positive Rate at various classification thresholds. An AUC-ROC of ~0.81 confirms that the model has learned genuine discriminative patterns — a random classifier would score 0.50.

### 5.5 Interpretation

- **Recall of ~0.72** on the Yes class is the most critical metric in this application. It means the model correctly identifies approximately 72% of actual cloudburst events. In a disaster prediction context, a False Negative (missed cloudburst) is far more dangerous than a False Positive (false alarm): the cost of a missed cloudburst warning — potential loss of life — vastly exceeds the cost of an unnecessary evacuation notice.
- **Precision of ~0.52** means roughly half of all cloudburst predictions are correct. The lower precision is an acceptable trade-off for higher recall.
- **AUC-ROC of ~0.81** indicates that the model ranks a randomly chosen cloudburst day above a randomly chosen non-event day 81% of the time — a strong signal for a linear model on this type of data.

The primary limitation is that Logistic Regression is a linear classifier. Meteorological systems involve complex, non-linear interactions between features (e.g., the joint effect of high humidity and falling pressure is likely multiplicative, not merely additive). This motivates the tree-based models in Phase 4.

---

## 6. Model Optimisation & Unsupervised Learning

### 6.1 Overfitting / Underfitting Analysis

Before optimising, it is essential to diagnose whether the baseline model is overfitting (memorising training data) or underfitting (failing to capture the underlying patterns). I compared the **F1 Score on the training set vs. the test set** for the Logistic Regression model.

As expected, the training and test F1 scores were very close for Logistic Regression — both around 0.60. This indicates **slight underfitting** rather than overfitting: the model's linear decision boundary cannot fully capture the non-linear relationships in the meteorological feature space. The gap between train and test performance is small (well under 0.10), so overfitting is not a concern for LR. However, the absolute values are modest, which motivates moving to more expressive model families.

This diagnosis is important: applying heavy regularisation to an already underfitting model would worsen performance. Instead, the right strategy is to allow more model complexity — motivating the Decision Tree and Random Forest in Phase 4.

### 6.2 Regularisation & Feature Scaling

`StandardScaler` was applied to all numerical features before refitting Logistic Regression. Scaling ensures that gradient convergence is not distorted by features with very different magnitudes. After scaling, I swept over regularisation strengths: C ∈ {0.01, 0.1, 1, 10}, where C is the inverse of regularisation strength (lower C = stronger L2 penalty).

The best-performing configuration used **C = 1** (the default), achieving a marginal but consistent improvement:

| Metric | Baseline LR | Optimised LR |
|---|---|---|
| Accuracy | ~0.76 | ~0.77 |
| F1 Score | ~0.60 | ~0.61 |
| AUC-ROC | ~0.81 | ~0.82 |

The marginal gain confirms that Logistic Regression is near its ceiling on this dataset. The limiting factor is model expressiveness, not scale or regularisation.

### 6.3 Principal Component Analysis (PCA)

PCA was applied to the StandardScaled training data to explore dimensionality reduction and understand feature redundancy. The cumulative explained variance curve (Figure: `plot7_pca_variance.png`) showed that approximately **90% of variance** is captured in roughly 10–12 principal components out of ~18 total features. This confirms that the feature space contains significant redundancy — particularly from the near-perfectly correlated temperature and pressure pairs identified in the heatmap.

A 2D PCA scatter plot (Figure: `plot8_pca_scatter.png`) coloured by target class showed **partial but meaningful class separation** along the first two principal components. The two classes are not cleanly separable in this 2D projection — there is substantial overlap — but their centres of mass are distinct. This tells us three things:

1. The features carry genuine discriminative signal (the classes are not randomly mixed).
2. A linear boundary in the original feature space is insufficient for clean separation.
3. Non-linear classifiers with access to the full feature space should perform significantly better.

### 6.4 K-Means Clustering

K-Means clustering was applied to the PCA-reduced data to discover natural groupings of weather conditions. The **elbow method** (Figure: `plot9_kmeans_elbow.png`) plotted the within-cluster sum of squared errors (WCSSE) against K from 2 to 10. The slope notably decreased at **K = 4**, indicating that four clusters represent a natural partitioning of the data.

The four clusters were projected into the 2D PCA space (Figure: `plot10_kmeans_clusters.png`), and the **cloudburst rate per cluster** was computed (proportion of records within each cluster labelled Yes = 1). The result showed that certain clusters had cloudburst rates 2–3× higher than others. Qualitatively, these high-risk clusters corresponded to weather conditions characterised by high humidity, low pressure, and strong winds — the same features identified as important by the EDA boxplots.

This unsupervised finding is independently significant: it validates the feature set's predictive value without relying on the supervised label. Natural weather pattern groupings (identified purely from feature similarity) align with cloudburst occurrence, providing additional evidence that the chosen features encode genuine meteorological signal.

---

## 7. Advanced Models

### 7.1 Decision Tree

Decision Trees partition the feature space by recursively selecting the feature and threshold that best separates the classes. They are inherently non-linear and can capture the interaction effects that Logistic Regression cannot.

#### Initial (Unpruned) Tree

An unpruned Decision Tree was trained first as a baseline. It achieved very high training F1 (~0.95+) but significantly lower test F1, indicating **severe overfitting**. The tree memorised noise in the training data rather than generalising. This is the classic behaviour of unconstrained Decision Trees and necessitates pruning via hyperparameter tuning.

#### After Tuning (GridSearchCV)

Hyperparameter tuning was performed using `GridSearchCV` with 5-fold cross-validation. The parameter grid was:

| Parameter | Values Tested |
|---|---|
| `max_depth` | 5, 10, 15, 20 |
| `min_samples_split` | 10, 20, 50 |
| `min_samples_leaf` | 5, 10, 20 |

After tuning, the overfitting gap narrowed considerably and test performance improved meaningfully. Constraining tree depth reduces the model's ability to memorise individual training samples.

### 7.2 Random Forest

Random Forest is an ensemble of Decision Trees trained on bootstrapped subsets of the training data, with feature randomness added at each split. By averaging predictions across many trees trained on different data subsets, the model dramatically reduces variance (overfitting) while maintaining the ability to capture non-linear interactions.

Random Forest is particularly well-suited for this dataset because:
- It handles the mix of numerical and encoded categorical features naturally.
- It is robust to irrelevant or noisy features (they rarely get selected at splits).
- Feature importance can be extracted, providing interpretability alongside strong performance.

### 7.3 Hyperparameter Tuning for Random Forest

GridSearchCV was applied to Random Forest with the following grid:

| Parameter | Values Tested |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 10, 20, None |
| `min_samples_split` | 10, 20 |

The best configuration used `n_estimators=200`, `max_depth=20`, and `min_samples_split=10` in most runs. The tuned Random Forest generalised well to the test set without overfitting.

**Feature Importance** (Figure: `plot11_feature_importance.png`) ranked features by their average contribution to reducing impurity across all trees:

| Rank | Feature |
|---|---|
| 1 | Humidity3pm |
| 2 | Pressure3pm |
| 3 | WindGustSpeed |
| 4 | Rainfall |
| 5 | Cloud3pm |
| 6 | Humidity9am |
| 7 | Temperature3pm |
| 8 | Pressure9am |
| 9 | WindSpeed3pm |
| 10 | Cloud9am |

This ranking is entirely consistent with the EDA findings from Phase 1 (boxplots showing Humidity3pm and Pressure3pm as the strongest discriminators) and with established meteorological knowledge about cloudburst precursors. The alignment between data-driven feature importance and domain knowledge is a strong indicator that the model is learning genuine patterns rather than noise.

---

## 8. Model Comparison & Final Selection

The four models trained throughout this project were compared on the test set across five metrics:

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | ~0.76 | ~0.52 | ~0.72 | ~0.60 | ~0.81 |
| Logistic Regression (Optimised) | ~0.77 | ~0.53 | ~0.73 | ~0.61 | ~0.82 |
| Decision Tree (Tuned) | ~0.78 | ~0.55 | ~0.70 | ~0.62 | ~0.83 |
| **Random Forest (Tuned)** | **~0.82** | **~0.60** | **~0.75** | **~0.67** | **~0.88** |

*(Exact values depend on execution environment; refer to Figure `plot13_model_comparison.png` for bar chart and Figure `plot12_roc_all_models.png` for ROC curves.)*

### Final Model Selection: Tuned Random Forest

The **Tuned Random Forest** is selected as the final model based on the following justification:

1. **Highest F1 Score (~0.67):** F1 is the primary metric given class imbalance; the Random Forest outperforms all other models by a clear margin over the Decision Tree (~0.62) and Logistic Regression variants (~0.60–0.61).

2. **Highest AUC-ROC (~0.88):** An AUC-ROC of 0.88 indicates that the model ranks a random cloudburst day above a random non-event day 88% of the time — substantially better than the Logistic Regression baseline (0.81).

3. **Best Recall on the minority class (~0.75):** In a disaster prediction context, recall on the Yes class is the most operationally critical metric. The Random Forest correctly identifies 75% of actual cloudburst events, missing only 1 in 4 — compared to 28% missed by the Decision Tree (recall ~0.70).

4. **Ensemble robustness:** Unlike the Decision Tree, which remains sensitive to individual training examples even after pruning, the Random Forest's averaging mechanism makes it inherently more stable and less prone to overfitting. The train vs. test performance gap is small, confirming generalisation.

5. **Interpretability via feature importance:** Despite being an ensemble, the Random Forest provides a clear ranking of feature importance that aligns with domain knowledge, making its decisions defensible to a meteorological expert.

The final model's confusion matrix (Figure: `plot14_confusion_matrix_final.png`) shows a meaningful reduction in False Negatives (missed cloudbursts) compared to the baseline, which is the highest-priority operational goal.

---

## 9. Conclusions

This project demonstrated a complete, end-to-end Machine Learning workflow for a real-world binary classification problem: predicting the next-day occurrence of a cloudburst from current meteorological observations.

**Key findings and achievements:**

- **Preprocessing decisions** were each grounded in clear reasoning — dropping leaky and high-missingness features, using robust median/mode imputation, and applying balanced class weights to address imbalance without synthetic data generation.

- **EDA** revealed physically meaningful patterns: afternoon humidity, falling pressure, and wind gust speed are the strongest discriminators between cloudburst and non-event days, consistent with established atmospheric science.

- **PCA** confirmed significant feature redundancy (temperature and pressure pairs) and partial class separability, motivating non-linear models.

- **K-Means clustering** identified natural weather regime groupings that independently correlated with cloudburst occurrence, validating the feature set without supervision.

- **Model progression** from Logistic Regression (AUC ~0.81, F1 ~0.60) to Tuned Random Forest (AUC ~0.88, F1 ~0.67) demonstrated clear, justified performance improvements at each step.

- **Limitations acknowledged:** The dataset reflects Australian weather patterns, which differ from Indian orographic cloudburst dynamics. The model treats each day independently, ignoring temporal autocorrelation. Real deployment would require integration with a live weather data feed.

Despite these limitations, the Tuned Random Forest represents a functional, interpretable, and deployable system that can meaningfully assist in early cloudburst warning.

---

## 10. Future Scope

Several concrete directions could extend and improve this work:

1. **Time-Series Modelling (LSTM/GRU):** This project treats each day independently. In reality, atmospheric conditions evolve: today's pressure trend is more informative than today's absolute pressure. Sequence models such as Long Short-Term Memory (LSTM) or Gated Recurrent Unit (GRU) networks could capture multi-day temporal dependencies and are likely to improve prediction significantly.

2. **Indian Meteorological Data Integration:** Cloudbursts in India are predominantly orographic — driven by moist air currents interacting with the Himalayan and Western Ghats terrain. Integrating data from the India Meteorological Department (IMD) with the addition of geographic features (elevation, slope aspect, proximity to mountain ridgelines) would make the system directly applicable to the most cloudburst-affected regions of India.

3. **Gradient Boosting (XGBoost / LightGBM):** Gradient-boosted trees frequently outperform Random Forests on tabular data by sequentially correcting residual errors. Benchmarking XGBoost and LightGBM against the current best model is a natural next step.

4. **Deployment as a Web Application:** The trained model could be serialised (via `joblib` or `pickle`) and served through a REST API (Flask / FastAPI) integrated with a real-time weather API such as OpenWeatherMap. A simple web interface could display next-day cloudburst risk probability for any queried location.

5. **Multi-Class Severity Prediction:** Rather than binary (cloudburst / no cloudburst), future work could predict severity levels — mild, moderate, and severe cloudbursts — enabling more graduated and proportionate emergency responses.

6. **Explainability (SHAP Values):** Deploying SHAP (SHapley Additive exPlanations) analysis on the Random Forest would provide instance-level explanations: for each prediction, SHAP would show exactly which features pushed the risk score up or down, making the system more transparent and trustworthy for operational use by disaster management authorities.

---

## 11. References

1. **Dataset:** Akshat, "CloudBurst Prediction Dataset," Kaggle, 2023. Available: https://www.kaggle.com/datasets/akshat234/cloudburst

2. **Breiman, L.** (2001). "Random Forests." *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

3. **Pedregosa, F. et al.** (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html

4. **Hochreiter, S. & Schmidhuber, J.** (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

5. **IMD (India Meteorological Department):** National Weather Services, Government of India. Available: https://mausam.imd.gov.in

6. **Jolliffe, I. T.** (2002). *Principal Component Analysis*, 2nd ed. Springer. — Referenced for PCA theory and interpretation.
