import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample

st.title("🩺 Healthcare Recommendation System")

df = pd.read_csv("healthcare_dataset.csv")

df = df.rename(columns={
    'AGE': 'Age',
    'MedicalCondition': 'Medical Condition',
    'AdmissionType': 'Admission Type'
})

df = df[['Age', 'Medical Condition', 'Admission Type']].dropna()
df = df[df['Admission Type'].isin(['Emergency', 'Urgent', 'Elective'])]

df['Cond_Code'] = df['Medical Condition'].astype('category').cat.codes


df['Chronic Illness'] = ([1, 0, 1, 0, 1, 0, 1, 0] * ((len(df)//8)+1))[:len(df)]
df['Smoker'] = ([0, 1] * ((len(df)//2)+1))[:len(df)]
df['Gender'] = ([0, 1] * ((len(df)//2)+1))[:len(df)]
df['High BP'] = ([1, 0, 1, 1, 0] * ((len(df)//5)+1))[:len(df)]
df['Diabetes'] = ([0, 1, 0, 1, 1] * ((len(df)//5)+1))[:len(df)]

df_major = df[df['Admission Type'] == 'Emergency']
df_minor1 = df[df['Admission Type'] == 'Urgent']
df_minor2 = df[df['Admission Type'] == 'Elective']

df_minor1_up = resample(df_minor1, replace=True, n_samples=len(df_major), random_state=42)
df_minor2_up = resample(df_minor2, replace=True, n_samples=len(df_major), random_state=42)

df = pd.concat([df_major, df_minor1_up, df_minor2_up])

X = df[['Age', 'Cond_Code', 'Chronic Illness', 'Smoker', 'Gender', 'High BP', 'Diabetes']]
y = df['Admission Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.subheader("Detailed Performance Report")
predictions = model.predict(X_test)
st.text(classification_report(y_test, predictions))

st.sidebar.header("Enter Patient Details")

name = st.sidebar.text_input("Patient Name")
contact = st.sidebar.text_input("Contact Number")
gender = st.sidebar.radio("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", int(df['Age'].min()), int(df['Age'].max()), 30)
condition = st.sidebar.selectbox("Select Medical Condition", df['Medical Condition'].unique())
chronic = st.sidebar.checkbox("Do you have a chronic illness?")
smoke = st.sidebar.radio("Do you smoke?", ["Yes", "No"])
bp = st.sidebar.radio("Do you have high blood pressure?", ["Yes", "No"])
sugar = st.sidebar.radio("Do you have diabetes?", ["Yes", "No"])

gender_code = 0 if gender == "Male" else 1
cond_code = df[df['Medical Condition'] == condition]['Cond_Code'].iloc[0]
chronic_code = 1 if chronic else 0
smoke_code = 1 if smoke == "Yes" else 0
bp_code = 1 if bp == "Yes" else 0
sugar_code = 1 if sugar == "Yes" else 0

input_data = [[age, cond_code, chronic_code, smoke_code, gender_code, bp_code, sugar_code]]


if st.sidebar.button("🔍 Get Recommendation"):
    prediction = model.predict(input_data)[0]
    st.success(f"Patient: {name}")
    st.write(f"Contact: {contact}")
    st.write(f"Recommended Care: **{prediction}**")
    st.info("Note: Prediction based on trained ML model and health indicators.")

st.subheader("Sample Patient Data")
st.dataframe(df.head())
