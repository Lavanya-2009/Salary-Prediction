💼 Employee Salary Prediction & Workforce Analytics System

An end-to-end Machine Learning project that predicts employee salaries using explainable AI techniques and provides workforce segmentation insights.

Built using Scikit-Learn, SHAP, Streamlit, and KMeans Clustering.

📌 1. Problem Statement

In many organizations, determining fair and competitive salaries is challenging due to multiple influencing factors such as:

Experience

Education Level

Industry

Work Mode

Skills

Working Hours

Manual estimation is inconsistent and biased.

🎯 Objective

Build an intelligent system that:

Predicts employee salary accurately

Explains why a salary is predicted (Explainable AI)

Identifies important salary-driving factors

Segments employees into meaningful groups

Provides an interactive dashboard for decision making

📊 2. Dataset Description

File: employee_salary_dataset.csv

The dataset contains structured employee data with the following features:

Feature	Description
Age	Employee age
Education Level	Bachelor's / Master's / PhD
Job Title	Employee designation
Experience (Years)	Total years of experience
Industry	IT / Finance / Healthcare / etc
Hours/Week	Average working hours
Work Mode	Remote / Hybrid / Onsite
Skills	Key skill tags
Salary	Target variable (Annual Salary)
🔎 Target Variable

Salary – Continuous numeric value representing annual compensation.

🏗 3. Project Architecture
Dataset → Preprocessing → ML Pipeline → Model Training
        → Prediction → Explainability (SHAP)
        → Workforce Segmentation (KMeans)
        → Streamlit Dashboard
⚙️ 4. Machine Learning Pipeline

The project uses Scikit-Learn Pipeline for clean and production-ready modeling.

🔹 Preprocessing

OneHotEncoding for categorical features

Standard scaling for numeric features

Combined using ColumnTransformer

🔹 Model Used

Random Forest Regressor (Salary Prediction)

KMeans Clustering (Employee Segmentation)

🔹 Model Storage

Trained models are saved in:

models/salary_model.pkl
models/kmeans_model.pkl
🤖 5. Explainable AI (SHAP)

To improve transparency, SHAP (SHapley Additive Explanations) is used.

The system explains:

Which features increased salary

Which features decreased salary

Top influencing factors for each prediction

This makes the model:

Interpretable

Business-friendly

Recruiter-ready

🚀 6. Streamlit Web Application

The project includes a fully interactive Streamlit dashboard (app.py).

🔹 Key Features
✅ 1. Salary Prediction

User inputs:

Age

Education

Experience

Industry

Skills

Work Mode

Hours per week

The model predicts annual salary instantly.

✅ 2. Feature Importance Visualization

Displays which features generally impact salary the most.

✅ 3. SHAP Explanation

Shows top 10 features affecting the specific prediction.

✅ 4. Employee Segmentation

Uses KMeans clustering to group employees into salary-based segments.

✅ 5. Interactive UI

Built using Streamlit with:

Sliders

Dropdowns

Dynamic charts

Clean layout

📸 7. Screenshots Section (Add After Taking Screenshots)

After running the app, take screenshots and add inside a screenshots/ folder.

<img width="1920" height="1020" alt="Screenshot 2026-02-24 182054" src="https://github.com/user-attachments/assets/50703e23-1af9-49df-8a3f-e13b5927037b" />

<img width="1920" height="1020" alt="Screenshot 2026-02-24 182003" src="https://github.com/user-attachments/assets/fdd99cfc-2ff8-4f91-8306-c6b62742d19a" />



📂 8. Project Folder Structure
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
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
🛠 9. Tech Stack
Category	Technology
Programming	Python
ML Library	Scikit-Learn
Explainability	SHAP
Data Handling	Pandas, NumPy
Visualization	Matplotlib
Web App	Streamlit
Model Persistence	Pickle
▶️ 10. How To Run The Project
Step 1: Clone Repository
git clone https://github.com/YOUR_USERNAME/employee-salary-prediction.git
cd employee-salary-prediction
Step 2: Create Virtual Environment (Optional)
conda create -n salary_env python=3.10
conda activate salary_env
Step 3: Install Requirements
pip install -r requirements.txt
Step 4: Run Streamlit App
streamlit run app.py

App will open at:

http://localhost:8501
📈 11. Model Performance

Uses Random Forest Regressor

Handles categorical + numerical data

Robust to overfitting

Good generalization

(You can add R² score from notebook here for extra strength.)

🌟 12. Key Highlights (Resume Worthy Points)

Built end-to-end ML pipeline

Integrated Explainable AI (SHAP)

Implemented employee clustering

Developed interactive dashboard

Saved and deployed trained models

Structured production-ready project

🔮 13. Future Improvements

Deploy on Streamlit Cloud

Add salary benchmarking comparison

Add PDF report download

Integrate real HR dataset

Add role-based salary prediction

👩‍💻 Author

Lavanya
Machine Learning & Data Science Enthusiast

🏁 Conclusion

This project demonstrates:

Strong ML fundamentals

Explainable AI implementation

End-to-end system design

Production-style deployment

Business-oriented thinking
