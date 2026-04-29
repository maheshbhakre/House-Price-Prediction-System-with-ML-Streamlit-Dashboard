# 🏠 House Price Prediction System using Machine Learning

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data--Analysis-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project implements an **end-to-end house price prediction system** that:

- 🏠 Predicts property prices using Machine Learning  
- 📊 Performs data analysis and feature engineering  
- 📈 Visualizes price trends and feature impact  
- 🖥 Provides an interactive Streamlit dashboard  

---

## 📊 Dashboard Preview

![Dashboard](images/dash1.png)  
![Dashboard](images/dash2.png)

---

## 🛠 Problem Statement

Real estate pricing is challenging due to:

- ❌ Manual estimation errors  
- ❌ Lack of data-driven insights  
- ❌ Inconsistent pricing across locations  

---

## ✅ Solution

This system provides:

- Accurate price prediction using ML  
- Feature-based valuation insights  
- Interactive dashboard for user input  
- Visual analytics for better decision-making  

---

## 🏭 Industry Relevance

| Industry | Application |
|--------------------|--------------------------------|
| Real Estate | Property price estimation |
| Banking | Loan collateral valuation |
| Investment Firms | ROI analysis |
| Property Portals | Listing price suggestions |
| PropTech | Automated valuation models |

---

## 📊 Business Impact

- 📈 Better pricing decisions  
- 💰 Reduced under/overpricing  
- ⚡ Faster property evaluation  
- 📊 Data-driven insights  

---

## ⚙ Tech Stack

- **Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn (Linear Regression, Random Forest)  
- **Visualization:** Matplotlib, Seaborn  
- **Dashboard:** Streamlit  
- **Model Storage:** Joblib  

---

## 📊 Dataset

Synthetic Housing Dataset (CSV)

### Features:

- area  
- bedrooms  
- bathrooms  
- floors  
- age  
- parking  
- furnishing  
- location  

### Target:

- `price` → house price prediction  

---

## 🏗 System Architecture

```
Raw Data → Data Cleaning → Feature Engineering → Model Training → Evaluation → Prediction → Dashboard
```

---

## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   ├── houses.csv
│   ├── houses_clean.csv
│   ├── houses_featured.csv
│
├── notebooks/
│   └── eda.py
│
├── src/
│   ├── data_generation.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── visualization.py
│   └── predict.py
│
├── models/
│   └── house_price_model.pkl
│
├── outputs/
│   ├── error_analysis.csv
│   └── feature_importance.csv
│
├── images/
│   ├── heatmap.png
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── (other graphs)
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙ Installation & Setup

```
git clone https://github.com/maheshbhakre/house-price-prediction-ml.git
cd house-price-prediction-ml

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🖥 Usage

```
python src/model_training.py
python src/model_evaluation.py
streamlit run app.py
```

---

## 📊 Model Performance

- Random Forest performed better than Linear Regression  
- Lower RMSE and higher R² score  

---

## 📸 PHASE-WISE IMPLEMENTATION PROOF

### 🔹 Phase 1
![P1](images/phase1.png)

### 🔹 Phase 2
![P2](images/phase2.png)  
![P2](images/phase2.2.png)

### 🔹 Phase 3
![P3](images/phase3.png)  
![P3](images/phase3.2.png)

### 🔹 Phase 4
![P4](images/phase4.1.png)  
![P4](images/phase4.2.png)

### 🔹 Phase 5
![P5](images/phase5.1.png)  
![P5](images/phase5.2.png)

### 🔹 Phase 6
![P6](images/phase6.1.png)

### 🔹 Phase 7
![P7](images/phase7.1.png)  
![P7](images/phase7.2.png)

### 🔹 Phase 8
![P8](images/phase8.1.png)

### 🔹 Phase 9
![P9](images/phase9.1.png)  
![P9](images/phase9.2.png)

---

## 📊 Additional Visualizations

![Heatmap](images/heatmap.png)  
![Feature Importance](images/feature_importance.png)  
![Bedrooms vs Price](images/bedrooms_vs_price.png)  
![Parking vs Price](images/parking_vs_price.png)  
![House Age Score vs Price](images/house_age_score_vs_price.png)  
![Price Distribution](images/price_distribution.png)  
![Residuals](images/residuals.png)  

---

## 👨‍💻 Author

Mahesh Bhakre  

---

## 🌐 CONNECT WITH ME  
<a href="https://github.com/maheshbhakre">
<img src="https://img.shields.io/badge/GitHub-Profile-black?style=for-the-badge&logo=github">
</a>

<a href="https://www.linkedin.com/in/maheshbhakreds1242">
<img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin">
</a>

<a href="https://www.instagram.com/mahesh_bhakre__2k06">
<img src="https://img.shields.io/badge/Instagram-Follow-purple?style=for-the-badge&logo=instagram">
</a>

<a href="https://saimfsd.github.io/mahesh-portfolio/">
<img src="https://img.shields.io/badge/Portfolio-Visit%20Website-orange?style=for-the-badge&logo=google-chrome">
</a>
---

## ⭐ NOTE

This project demonstrates a complete **end-to-end machine learning pipeline for house price prediction**, including data preprocessing, feature engineering, model training, evaluation, and dashboard deployment.
