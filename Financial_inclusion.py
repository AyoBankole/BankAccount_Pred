import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv('Financial_inclusion_dataset.csv')
data.drop('uniqueid', axis = 1, inplace = True)

df = data.copy()

st.markdown("<h1 style='color: #12372A; text-align: center; font-family: Sans serif'>BANK ACCOUNT PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin: -30px; color: #561C24; text-align: center; font-family: cursive'>Built By The Mushin Data Guy</h4>", unsafe_allow_html=True)
st.image('bank.png', width=350, use_column_width=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: cursive'>Project Overview</h4>", unsafe_allow_html=True)
st.markdown("<p style='text-align: justify'>The bank account opening prediction modeling project aims to leverage machine learning techniques to develop an accurate and robust model capable of predicting whether a customer opens or owns a bank account or not. By analyzing historical data, identifying key features influencing the customer's decision, and employing advanced classification algorithms, the project seeks to provide valuable insights for business analysts, entrepreneurs, large and small scale businesses. The primary objective of this project is to create a reliable machine learning model that accurately predicts customer's decision based on relevant features such as age, household size, type of job, and other influencing factors. The model should be versatile enough to adapt to different business plans, providing meaningful predictions for a wide range of businesses and sectors.</p>", unsafe_allow_html=True)
st.sidebar.image('pngwing.com.png', width=150, use_column_width=True, caption='Welcome User')
st.markdown("<br>", unsafe_allow_html=True)


# scaler = StandardScaler()

# df.drop(['uniqueid'], axis=1, inplace=True)

encoders = {}

for i in data.select_dtypes(exclude = 'number').columns:
    encoder = LabelEncoder()
    df[i] = encoder.fit_transform(df[i])
    encoders[i + '_encoder'] = encoder



x = df.drop('bank_account', axis=1)
y = df.bank_account

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, stratify=y)

model = LogisticRegression()
model.fit(xtrain, ytrain)

st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: cursive;'>PREDICTOR MODEL</h4>", unsafe_allow_html=True)
# st.dataframe(data)


country = st.sidebar.selectbox('COUNTRY OF REGION', data['country'].unique())
year = st.sidebar.number_input('YEAR', data['year'].min(), data['year'].max())
location_type = st.sidebar.selectbox('TYPE OF LOCATION', data['location_type'].unique())
cellphone_access = st.sidebar.selectbox('ACCESS TO TELEPHONE', data.cellphone_access.unique())
household_size = st.sidebar.number_input('HOUSEHOLD SIZE', data['household_size'].min(), data['household_size'].max())
gender_of_respondent = st.sidebar.selectbox('GENDER', data.gender_of_respondent.unique())
age_of_respondent = st.sidebar.number_input('RESPONDENT AGE', data['age_of_respondent'].min(), data['age_of_respondent'].max())
relationship_with_head = st.sidebar.selectbox('RELATIONSHIP WITH THE HEAD OF THE HOUSE', data.relationship_with_head.unique())
marital_status = st.sidebar.selectbox('MARITAL STATUS', data.marital_status.unique())
educational_level = st.sidebar.selectbox('HIGHEST EDUCATION LEVEL', data['education_level'].unique())
job_type = st.sidebar.selectbox('JOB TYPE', data.job_type.unique())

new_country = encoders['country_encoder'].transform([country])
new_location_type = encoders['location_type_encoder'].transform([location_type])
new_cellphone_access = encoders['cellphone_access_encoder'].transform([cellphone_access])
new_gender_of_respondent = encoders['gender_of_respondent_encoder'].transform([gender_of_respondent])
new_relationship_with_head = encoders['relationship_with_head_encoder'].transform([relationship_with_head])
new_marital_status = encoders['marital_status_encoder'].transform([marital_status])
new_educational_level = encoders['education_level_encoder'].transform([educational_level])
new_job_type = encoders['job_type_encoder'].transform([job_type])


input_var = pd.DataFrame({
    'country': [new_country],
    'year': [year],
    'location_type': [new_location_type],
    'cellphone_access': [new_cellphone_access],
    'household_size': [household_size],
    'age_of_respondent': [age_of_respondent],
    'gender_of_respondent': [new_gender_of_respondent],
    'relationship_with_head': [new_relationship_with_head],
    'marital_status': [new_marital_status],
    'education_level': [new_educational_level],
    'job_type': [new_job_type]
})



st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h5 style='margin: -30px; color: olive; font:sans-serif' >", unsafe_allow_html=True)
st.dataframe(input_var)

# Check column order and feature names
#print("Input_var columns:", input_var.columns)
#print("Model training columns:", xtrain.columns)



prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred:
        # Include the prediction step here
        predicted = model.predict(input_var)
        output = 'NOT HAVING A BANK ACCOUNT' if predicted[0] == 0 else 'HAVING A BANK ACCOUNT'
        st.success(f'The individual is predicted to {output}')
        st.balloons()
with interprete:
    st.header('The Interpretation Of The Model')
    st.write("In summary, the model achieves an overall accuracy of 77%, indicating a reasonable performance in predicting which individuals are most likely to have or use a bank account. Precision, recall, and F1-score metrics provide insights into the model's effectiveness for each class, and the support values give context to the distribution of instances in the dataset.")

