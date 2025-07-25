# 💼 Employee Salary Prediction App

This web app predicts whether an employee earns **more than $50K or less than or equal to $50K**, based on various demographic and work-related features.

## 📊 Input Features
- **Age**: Employee's age (18–75)
- **Workclass**: Type of employment sector (Private, Government, Self-employed, etc.)
- **Education Level**: Highest level of education completed
- **Marital Status**: Current marital status
- **Occupation**: Primary job role
- **Hours per Week**: Number of hours worked weekly

## 🧠 Model Info
- Trained with `MLPClassifier` from `scikit-learn`
- Categorical features encoded using `LabelEncoder`
- Input scaled using `MinMaxScaler`
- Achieved high accuracy on clean, preprocessed dataset

## 🚀 Technologies Used
- `Gradio` for interactive web UI
- `scikit-learn` for training and modeling
- `pandas` and `joblib` for data handling and model storage

---

## 🚀 Live Demo

Try the live app here: [Employee Salary Prediction on Hugging Face](https://huggingface.co/spaces/VishaalR/Employee-Salary-Prediction)

---

Built by Vishaal R
