# 🚗 AutoPricerAI
### Pre-owned Vehicle Intelligence Platform
**Built for ForgePoint AI · Internship Project 2026**

---

## 🎯 Problem Statement
Pre-owned car dealers need to quote accurate prices faster than competitors. 
Manual valuation is slow, inconsistent, and expertise-dependent. 
AutoPricerAI solves this by predicting the fair market price of any vehicle 
in under 2 seconds using machine learning.

---

## ✨ Features
- 📊 **Interactive Dashboard** — Market trends, brand analysis, price distribution, depreciation curves
- 🔍 **Data Quality Pipeline** — Automated cleaning, outlier removal, feature engineering
- 🤖 **Instant Price Predictor** — Enter car details, get best market quote immediately
- 📈 **Model Performance** — R² score, feature importance, model explainability
- 📱 **Mobile Ready** — Fully responsive, works on any device

---

## 🧠 ML Pipeline
- **Algorithm:** XGBoost Regressor with log-transform on target
- **Features:** Age, KM Driven, Brand, Fuel Type, Transmission, Engine CC, Max Power, Owner Count + 4 engineered features
- **Accuracy:** ~93% R² on CarDekho India dataset
- **Error:** Mean Absolute Error ~₹80,000

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Visualization | Plotly |
| ML Model | XGBoost |
| Data Processing | Pandas, NumPy |
| Deployment | Streamlit Cloud |

---

## 🚀 Run Locally
```bash
git clone https://github.com/YOURNAME/autopricerai.git
cd autopricerai
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
autopricerai/

│

├── app.py                 # Main Streamlit application

├── data_cleaning.py       # Data cleaning pipeline

├── model.py               # ML model training & prediction

├── requirements.txt       # Dependencies

├── model_artifacts.pkl    # Trained model (auto-generated)

└── data/

└── cardekho.csv       # Dataset

---

## 📊 Dataset
CarDekho India Pre-owned Vehicle Dataset
- 8,000+ listings
- 13 raw features
- Covers all major Indian brands

---

## 👥 Team
Built with ❤️ by Harshavardhana
Banglore, India · 2026

---

## 🏢 Client
**ForgePoint AI** · Industry Internship Project