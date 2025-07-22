import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

data_path = r'E:\Vishaal.R\Visual Studio Code\Python\adult 3.csv'
data = pd.read_csv(data_path)

data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)

categorical_cols = ['workclass', 'marital-status', 'occupation']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

drop_cols = ['education', 'relationship', 'race', 'capital-gain', 'capital-loss',
             'native-country', 'gender', 'fnlwgt']
data.drop(columns=[col for col in drop_cols if col in data.columns], inplace=True)

data = data[(data['age'] > 17) & (data['age'] <= 75)]

X = data.drop('income', axis=1)
y = data['income']

target_le = LabelEncoder()
y = target_le.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=1000,
                      early_stopping=True, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(encoders, 'encoders.joblib')
joblib.dump(target_le, 'target_encoder.joblib')

print("Training complete and files saved: model.joblib, scaler.joblib, encoders.joblib, target_encoder.joblib")
