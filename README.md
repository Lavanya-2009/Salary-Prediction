# 💼 Employee Salary Prediction & Workforce Analytics System

An end-to-end Machine Learning project that predicts employee salaries using Explainable AI techniques and provides workforce segmentation insights.

Built using Scikit-Learn, SHAP, Streamlit, and KMeans Clustering.

---

## 📌 Problem Statement

Determining fair and competitive employee salaries is challenging due to multiple influencing factors such as:

- Experience
- Education Level
- Industry
- Work Mode
- Skills
- Working Hours

Manual estimation can be inconsistent and biased.

### 🎯 Objective

Build an intelligent system that:

- Predicts employee salary accurately
- Explains why a salary is predicted (Explainable AI)
- Identifies key salary-driving factors
- Segments employees into meaningful groups
- Provides an interactive dashboard for decision-making

---

## 📊 Dataset Description

**File:** `employee_salary_dataset.csv`

The dataset contains structured employee data with the following features:

| Feature | Description |
|----------|-------------|
| Age | Employee age |
| Education Level | Bachelor's / Master's / PhD |
| Job Title | Employee designation |
| Experience (Years) | Total years of experience |
| Industry | IT / Finance / Healthcare / etc |
| Hours/Week | Average working hours |
| Work Mode | Remote / Hybrid / Onsite |
| Skills | Key skill tags |
| Salary | Target variable (Annual Salary) |

### 🎯 Target Variable
**Salary** – Continuous numeric value representing annual compensation.

---

## 🏗 Project Architecture

Dataset → Preprocessing → ML Pipeline → Model Training  
→ Prediction → Explainability (SHAP)  
→ Workforce Segmentation (KMeans)  
→ Streamlit Dashboard

---

## ⚙️ Machine Learning Pipeline

### 🔹 Data Preprocessing
- OneHotEncoding for categorical features
- Standard Scaling for numerical features
- Combined using ColumnTransformer
- Integrated into a Scikit-Learn Pipeline

### 🔹 Models Used
- **Random Forest Regressor** → Salary Prediction
- **KMeans Clustering** → Employee Segmentation

### 🔹 Model Storage
```
models/salary_model.pkl
models/kmeans_model.pkl
```

---

## 🚀 Streamlit Web Application

The project includes a fully interactive Streamlit dashboard (`app.py`).

### 🔹 Features

### ✅ 1. Salary Prediction
User inputs:
- Age
- Education
- Experience
- Industry
- Skills
- Work Mode
- Hours per week

The model predicts annual salary instantly.
<img width="1920" height="1020" alt="Screenshot 2026-02-24 182054" src="https://github.com/user-attachments/assets/f7d56392-bae2-4728-a7c9-7b0329569b68" />

---

### ✅ 2. Feature Importance Visualization
Displays overall most important salary-driving factors.

---




### ✅ 4. Employee Segmentation
Uses KMeans clustering to group employees into salary-based segments.
<img width="1920" height="1020" alt="Screenshot 2026-02-24 182003" src="https://github.com/user-attachments/assets/1e46ced7-792a-4ad8-a83d-b34933950eeb" />

---

### ✅ 5. Interactive UI
Built using:
- Sliders
- Dropdowns
- Dynamic charts
- Clean and intuitive layout

---

## 📂 Project Folder Structure

```
employee-salary-prediction/
│
├── app.py                         # Streamlit application
├── employee_salary_dataset.csv    # Dataset
├── ml_model..ipynb                # Model development notebook
│
├── models/
│   ├── salary_model.pkl           # Trained regression model
│   └── kmeans_model.pkl           # Clustering model
│
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## 🛠 Tech Stack

- Python
- Scikit-Learn
- SHAP
- Pandas
- NumPy
- Matplotlib
- Streamlit
- Pickle

---

## ▶️ How To Run The Project

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/employee-salary-prediction.git
cd employee-salary-prediction
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
conda create -n salary_env python=3.10
conda activate salary_env
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

App will open at:

```
http://localhost:8501
```

---

## 📈 Model Highlights

- Uses Random Forest Regressor
- Handles categorical + numerical features
- Integrated preprocessing pipeline
- Explainable AI support
- Production-style model saving

---

## 🌟 Key Highlights

- Built complete end-to-end ML pipeline
- Implemented Explainable AI (SHAP)
- Applied Employee Clustering
- Developed interactive dashboard
- Structured project for real-world deployment

---

## 🔮 Future Improvements

- Deploy on Streamlit Cloud
- Add downloadable PDF report
- Integrate real HR dataset
- Add salary benchmarking comparison
- Improve UI with advanced analytics

---

## 👩‍💻 Author

Lavanya  
Machine Learning & Data Science Enthusiast  

---

## 🏁 Conclusion

This project demonstrates:

- Strong Machine Learning fundamentals
- Explainable AI integration
- Business-focused problem solving
- Production-style project structuring
- Interactive deployment using Streamlit

---
