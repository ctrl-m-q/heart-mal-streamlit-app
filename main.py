import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from PIL import Image  # For image processing

# --- 1. Data Acquisition and Preprocessing ---

heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# metadata
metadata = heart_disease.metadata

# variable information
variables = heart_disease.variables

@st.cache_data  # Cache the data, so it doesn't need to be loaded every run.
def load_and_prepare_heart_disease_data(test_size=0.2, random_state=42):
    """
    Fetches the UCI Heart Disease dataset, prepares it, and splits it into training and testing sets.

    Args:
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Controls the shuffling applied to the data before splitting. Defaults to 42.

    Returns:
        tuple: A tuple containing (X_train, X_test, y_train, y_test, metadata, variables).
               X_train, X_test, y_train, y_test are pandas DataFrames representing the feature and target data.
               metadata and variables are dictionaries containing dataset metadata and variable information.
    """
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # metadata
    metadata = heart_disease.metadata

    # variable information
    variables = heart_disease.variables

    #X = X.dropna()
    #drop missing values with corresponding x and y
    # Drop rows with missing values from both X and y
    mask = X.isna().any(axis=1)  # create a mask of rows with any NaN values.
    X = X[~mask]  # apply the inverse mask to X.
    y = y[~mask]  # apply the inverse mask to y.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test, metadata, variables

# --- 2. Model Selection and Training ---
@st.cache_resource  # Cache the model.
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train.values.ravel()) #ravel to avoid data conversion warnings.
    return model

# ------Main Streamlit app iteration ------
def main():
    st.title("My Heart")
    page = st.sidebar.selectbox("Select page", ["Explore Data", "Clinical Insights", "Analyze my data"])

    X_train, X_test, y_train, y_test, metadata, variables = load_and_prepare_heart_disease_data()
    model = train_model(X_train, y_train) #loads the cached model.

    if page == "Explore Data":
        st.header("Explore Test Data:")
        data = pd.concat([X, y], axis=1)
        st.subheader("Dataset's Information:")
        st.write("Age - Patient's Age")
        st.write("Sex - Sex of the patient (encoded to 0 & 1; 1- Male, 0-Female)")
        st.write("CP - Chest pain (1- Typical Angina, 2 - Atypical Angina, 3 - Non anginal pain, 4- Asymptomatic)")
        st.write("trestbps - Resting blood pressure (mmHg)")
        st.write("chol - serum cholestoral (mmHg)")
        st.write("fbs - Fasting blood sugar > 120 mm/dl (1- true, 0 - false)")
        st.write(
            "restecg - resting electrocardiographic results (Value 0: normal, Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria)")
        st.write("thalach - Maximum heartrate achieved")
        st.write("exang - exercise induced angina (1 = yes; 0 = no)")
        st.write("oldpeak - ST depression induced by exercise relative to rest")
        st.write(
            "slope - the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)")
        st.write("ca - number of major vessels (0-3) colored by flourosopy")
        st.write("thal- 3 = normal; 6 = fixed defect; 7 = reversible defect")
        st.write(
            "Num - diagnosis of heart disease (angiographic disease status) (Value 0: < 50% diameter narrowing, Value 1: > 50% diameter narrowing (in any major vessel: attributes 59 through 68 are vessels))")
        st.subheader("*Data Stats*")
        st.write(data.describe())
        st.write(f"Training Dataset shape: {data.shape}")
        st.write("Data types:")
        dtypes_df= pd.DataFrame(data.dtypes, columns=['Data Type'])
        st.dataframe(dtypes_df)


        st.subheader("Visualise training Data")
        #distribution of numerical features
        num_feat = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cols_num = st.columns(min(2, len(num_feat)))
        for i , feat in enumerate(num_feat):
            fig = px.histogram(X, x=feat, title=f"Distribution of {feat}", height=300)
            cols_num[i % len(cols_num)].plotly_chart(fig, use_container_width=True)
            #st.plotly_chart(fig, use_container_width=True)



        fig_age_cp = px.scatter(data, x='age', y= 'cp', color='num', title='Age & Chest pain type correlation')
        st.plotly_chart(fig_age_cp, height = 400,  use_container_width=True)
        st.write("Age - Patient's Age")
        st.write("cp - Chest pain (1- Angina, 2- atypical angina, 3- non anginal pain, 4- asymptomatic ")
        st.write("num - diagnosis (0 - absence , 1 - <50% diameter narrowing, 2 - >50% diameter narrowing, 4- present")

        fig_age_thalach = px.scatter(data, x='age', y='thalach', color='num', title='Age & Thalach (max heart rate achieved)')
        st.plotly_chart(fig_age_thalach, use_container_width=True, height=400)
        st.write("Age - Patient's Age")
        st.write("thalach: Max heart rate achieved")
        st.write("num - diagnosis (0 - absence , 1 - <50% diameter narrowing, 2 - >50% diameter narrowing, 4- present")

        # Sex vs. trestbps
        fig_sex_trestbps = px.box(data, x='sex', y='trestbps', color='num', title='Sex and Resting Blood Pressure')
        st.plotly_chart(fig_sex_trestbps, use_container_width=True, height = 400)
        st.write("Sex - sex of patient (0 - Male, 1- Female)")
        st.write("trestbps - Resting blood pressure")


        st.subheader("Correlation Heatmap")
        corr = X.corr()
        fig = px.imshow(corr, title="correlation heatmap", aspect="Auto")
        st.plotly_chart(fig, use_container_width=True)

        st.header("Features vs Target")

        st.write("Features vs. Target (Box Plots)")
        cols_box = st.columns(min(3, len(num_feat))) #3 number of column
        for i, feature in enumerate(num_feat):
            fig = px.box(data, x='num', y=feature, title=f"{feature} vs. Target", height=300)
            cols_box[i % len(cols_box)].plotly_chart(fig, use_container_width =True)
            #st.plotly_chart(fig, use_container_width=True)


        st.subheader("Evaluate Model")
        y_pred = model.predict(X_test)
        st.write(classification_report(y_test, y_pred))

       # st.subheader("Confusion Matrix")
        #y_pred = model.predict(X_test)
        #cm = confusion_matrix(y_test, y_pred)


        #class_labels = sorted(list(set(y_test)))  # get the unique labels, and sort them.
        #fig_cm = px.imshow(cm,
                           #labels=dict(x="Predicted", y="Actual", color="Count"),
                           #x=class_labels,
                           #y=class_labels,
                           #text_auto=True)
        #st.plotly_chart(fig_cm)








    elif page == "Clinical Insights":
        st.header("Clinical Insights")

        st.subheader("Key Risk Factors")
        st.write("Key risk factor from Data includes Patient's Age, cholesterol, Chest pain type, blood pressure")
        st.write("*Please consult your medical practitioner for advice* ")

        st.subheader("Risk Assessment")
        age = st.slider("Age", 20, 90, 50)
        chol = st.slider("Cholesterol", 100, 600, 200)
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)

        if st.button("Predict Risk"):
            input_data = pd.DataFrame([[age, 1, cp, trestbps, chol, 0, 0, 150, 0, 1, 1, 0, 3]],
                                      columns=X_train.columns)  # example input.
            prediction = model.predict(input_data)[0]
            if prediction == 4:
                st.write("High Risk of Heart Disease")
            else:
                st.write("Low Risk of Heart Disease")

    elif page == "Analyze my data":
        st.write("Analyze my data")
        st.write("I have a saviour who intimately cares for my heart")

if __name__ == "__main__":
    main()
#Test commit
