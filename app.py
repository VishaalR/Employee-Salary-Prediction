import pandas as pd 
import joblib
import gradio as gr
import gradio.themes as themes
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, 'model.joblib'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
encoders = joblib.load(os.path.join(BASE_DIR, 'encoders.joblib'))

education_mapping = {
    "Preschool": 1, "Kindergarten": 2, "1st - 4th Grade": 3, "5th - 6th Grade": 4,
    "7th - 8th Grade": 5, "9th Grade": 6, "10th Grade": 7, "11th Grade": 8,
    "12th Grade": 9, "High School Graduate": 10, "Some College": 11,
    "Associate Degree (Academic)": 12, "Associate Degree (Vocational)": 13,
    "Bachelor's Degree": 14, "Master's Degree": 15,
    "Doctorate Degree (PhD)": 16, "Professional School Degree": 17
}
education_levels = list(education_mapping.keys())

marital_status_map = {
    "Married (civilian spouse)": "Married-civ-spouse",
    "Married (armed forces spouse)": "Married-AF-spouse",
    "Never married": "Never-married",
    "Divorced": "Divorced",
    "Widowed": "Widowed",
    "Separated": "Separated",
    "Married (spouse absent)": "Married-spouse-absent"
}
marital_status_options = list(marital_status_map.keys())

occupation_map = {
    "Tech Support": "Tech-support",
    "Craft Repair": "Craft-repair",
    "Other Service": "Other-service",
    "Sales": "Sales",
    "Executive Managerial": "Exec-managerial",
    "Professional Specialty": "Prof-specialty",
    "Handlers Cleaners": "Handlers-cleaners",
    "Machine Operator Inspector": "Machine-op-inspct",
    "Administrative Clerical": "Adm-clerical",
    "Farming Fishing": "Farming-fishing",
    "Transport Moving": "Transport-moving",
    "Private Household Service": "Priv-house-serv",
    "Protective Service": "Protective-serv",
    "Armed Forces": "Armed-Forces"
}
occupation_options = list(occupation_map.keys())

workclass_map = {
    "Private Sector": "Private",
    "Self-employed (Not Incorporated)": "Self-emp-not-inc",
    "Self-employed (Incorporated)": "Self-emp-inc",
    "Government - Federal": "Federal-gov",
    "Government - Local": "Local-gov",
    "Government - State": "State-gov",
    "Without Pay": "Without-pay"
}
workclass_options = list(workclass_map.keys())

def predict_salary(age, workclass_friendly, education_level, marital_status_friendly, occupation_friendly, hours_per_week):
    if workclass_friendly not in workclass_map:
        return f"❌ Error: Invalid workclass '{workclass_friendly}' selected."

    educational_num = education_mapping.get(education_level, 10)
    workclass = workclass_map[workclass_friendly]
    marital_status = marital_status_map[marital_status_friendly]
    occupation = occupation_map[occupation_friendly]

    try:
        input_dict = {
            'age': [age],
            'workclass': [encoders['workclass'].transform([workclass])[0]],
            'educational-num': [educational_num],
            'marital-status': [encoders['marital-status'].transform([marital_status])[0]],
            'occupation': [encoders['occupation'].transform([occupation])[0]],
            'hours-per-week': [hours_per_week]
        }
        input_df = pd.DataFrame(input_dict)
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]

        salary_class_map = {0: "<=50K", 1: ">50K"}
        pred_label = salary_class_map.get(pred, "Unknown")

        return f"🎯 Predicted Salary: {pred_label}"
    except Exception as e:
        return f"❌ Error: {e}"

theme = themes.Soft(primary_hue="blue", secondary_hue="purple")

iface = gr.Interface(
    fn=predict_salary,
    inputs=[
        gr.Number(label="🎂 Age", value=37, minimum=18, maximum=75, step=1),
        gr.Dropdown(choices=workclass_options, label="🏢 Workclass", value=workclass_options[0]),
        gr.Dropdown(choices=education_levels, label="🎓 Education Level", value="High School Graduate"),
        gr.Dropdown(choices=marital_status_options, label="💍 Marital Status", value=marital_status_options[0]),
        gr.Dropdown(choices=occupation_options, label="🛠️ Occupation", value=occupation_options[0]),
        gr.Slider(minimum=1, maximum=99, step=1, label="⏱️ Hours per Week", value=40),
    ],
    outputs="text",
    title="💼 Employee Salary Prediction",
    description=(
        "## 💼 Employee Salary Prediction App\n\n"
        "Welcome to the **Salary Predictor App**! 🌟\n\n"
        "Fill in the following fields to predict whether an employee earns <=50K or >50K.\n\n"
        "---\n"
        "### 📝 Input Information:\n"
        "- **Workclass**: Type of employment sector (e.g., Private, Government)\n\n"
        "- **Education**: Highest level of education completed\n\n"
        "- **Marital Status**: Current marital or relationship status\n\n"
        "- **Occupation**: Main job or role in the workforce\n\n"
        "- **Hours per Week**: Average number of hours worked weekly\n"
        "---"
    ),
    theme=theme
)

if __name__ == "__main__":
    iface.launch()
