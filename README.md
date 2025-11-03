# 🧬 Breast Cancer Diagnosis using Machine Learning

This project applies **machine learning** techniques to classify breast tumors as **benign** or **malignant** based on cell nucleus features.  
It uses the **Breast Cancer Wisconsin (Diagnostic)** dataset available in `scikit-learn`.

---

## 📖 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Next Steps](#next-steps)
- [References](#references)

---

## 🧠 Overview

Breast cancer is one of the most common cancers among women worldwide.  
Early and accurate diagnosis is critical for effective treatment.  
In this project, we train a **Random Forest Classifier** to predict whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on diagnostic measurements of cell nuclei.

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository – Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

- **Samples:** 569
- **Features:** 30 numeric features (e.g., radius, texture, smoothness)
- **Target Classes:**
  - `0` → Malignant
  - `1` → Benign

The dataset is also directly available through `scikit-learn`:
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
```

---

## 🗂 Project Structure

```plaintext
breast_cancer_diagnosis/
│
├── data/                     # (optional) if you export dataset manually
├── notebooks/
│   └── breast_cancer_diagnosis.ipynb
├── src/
│   └── model_training.py
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/<your-username>/breast_cancer_diagnosis.git
cd breast_cancer_diagnosis
pip install -r requirements.txt
```

**requirements.txt**
```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```
---

## 🔍 Exploratory Data Analysis

We explore:
- Class balance
- Feature distributions
- Correlations among numerical features

Example plot:
```python
sns.countplot(x='target', data=df)
plt.title("Class Distribution: 0 = Malignant, 1 = Benign")
```
---

## 🤖 Modeling
We trained a **Random Forest Classifier** with scaled features:

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
```

Evaluation metrics:
```python
from sklearn.metrics import classification_report, accuracy_score
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
```
---

## 📈 Results

| Metric | Score |
|---------|--------|
| **Accuracy** | 0.97 |
| **Precision (Malignant)** | 0.96 |
| **Recall (Malignant)** | 0.95 |
| **F1-Score (Malignant)** | 0.96 |

Confusion matrix:

![confusion-matrix](docs/confusion_matrix.png)

---

## 🧩 Feature Importance

Top predictive features identified by the Random Forest model:

| Feature | Importance |
|----------|-------------|
| worst concave points | 0.14 |
| mean concavity | 0.11 |
| mean radius | 0.09 |
| worst radius | 0.08 |
| mean perimeter | 0.07 |

Visualization:

```python
sns.barplot(x=feat_imp[:10], y=feat_imp.index[:10])
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
```

---

## 🚀 Next Steps
- Compare models (Logistic Regression, SVM, XGBoost)
- Apply PCA for dimensionality reduction
- Deploy interactive predictor using Streamlit
- Integrate SHAP for model explainability

---

## 📚 References

- UCI Breast Cancer Dataset
- scikit-learn Documentation
- Random Forest Algorithm


