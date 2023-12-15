
import pickle
import pandas as pd
import streamlit as st
import xgboost as xgb

from PIL import Image
months_dict = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}


data=pd.read_csv("https://raw.githubusercontent.com/YaseenIjaz/Customer-Conversion-Prediction/main/train%20(1).csv")
file_path="classifier.pkl"


with open(file_path, "rb") as pickle_in:
    loaded_classifier = pickle.load(pickle_in)


def main():
    # Create a page dropdown
    image = Image.open("logo.jpg")
    st.set_page_config(page_title="Insurance Prediction",page_icon='🧑‍⚕️',layout="wide",initial_sidebar_state="expanded")
    st.sidebar.image(image, width=100)
    st.sidebar.title("Insurance Prediction")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(":blue[NEW AGE INSURANCE]",anchor='center')
         
        
    with col2:
        st.image(image, width=250)
        

    page = st.sidebar.selectbox("Select One", ['ABOUT', "PREDICTION"])

    if page == "ABOUT":
        st.title(':grey[Welcome to Insurance Prediction]')
        
        st.header('About the Project:')
        st.write("This project is aimed at predicting insurance outcomes using advanced machine learning techniques.")
        
        st.header('Creator Profile:')
        st.write('**Creator:** Yaseen Ijaz')
        
        st.header('Technologies Used:')
        st.write('- Python')
        st.write('- Pandas')
        st.write('- NumPy')
        st.write('- Matplotlib')
        st.write('- Seaborn')
        st.write('- Streamlit for the Web App')
        st.write('- Machine Learning Libraries (e.g., scikit-learn, XGBoost)')
        
        st.header('How to Use:')
        st.write('Navigate to the "PREDICTION" page to make predictions for insurance outcomes'.)
        
    elif page == "PREDICTION":
        st.title('Insurance Prediction Page')
        st.write('This page allows you to make predictions for insurance outcomes.)
                 
    if page == "PREDICTION":
        st.title('PREDICTION')
        age_placeholder = st.text_input("Enter the Age", key="age_placeholder")
        age = int(age_placeholder) if age_placeholder else None

        job_options = ['Select an occupation'] + list(data.job.unique())
        job = st.selectbox("Occupation", job_options)

        if job == 'blue-collar':
            grouped=data[data['job']=='blue-collar']
            job = 0
        elif job == 'entrepreneur':
            grouped=data[data['job']=='entrepreneur']
            job = 1
        elif job == 'housemaid':
            grouped=data[data['job']=='housemaid']
            job = 2
        elif job == 'services':
            grouped=data[data['job']=='services']
            job = 3
        elif job == 'technician':
            grouped=data[data['job']=='technician']
            job = 4
        elif job == 'self-employed':
            grouped=data[data['job']=='self-employed']
            job = 5
        elif job == 'admin':
            grouped=data[data['job']=='admin']
            job = 6
        elif job == 'management':
            grouped=data[data['job']=='management']
            job=7
        elif job == 'unemployed':
            grouped=data[data['job']=='unemployed']
            job=8
        elif job == 'retired':
            grouped=data[data['job']=='retired']
            job=9
        elif job == 'student':
            grouped=data[data['job']=='student']
            job=10
        
        education_qual_options = ['Select an option'] + list(data.education_qual.unique())
        education_qual = st.selectbox("Education Qualification", education_qual_options)

        if education_qual == 'primary':
            grouped=data[data['education_qual']=='primary']
            education_qual = 0
        elif education_qual == 'secondary':
            grouped=data[data['education_qual']=='secondary']
            education_qual=1
        elif education_qual == 'tertiary':
            grouped=data[data['education_qual']=='tertiary']
            education_qual=2

        call_type_options = ['Select a call type'] + list(data.call_type.unique())
        call_type = st.selectbox("Call Type", call_type_options)

        if call_type == 'unknown':
            grouped=data[data['call_type']=='telephone']
            call_type = 0
        elif call_type == 'telephone':
            grouped=data[data['call_type']=='unknown']
            call_type=1
        elif call_type == 'cellular':
            grouped=data[data['call_type']=='cellular']
            call_type=2

        day = st.number_input("Call Day", min_value=int(data.day.min()), max_value=int(data.day.max()))
        
        mon = st.slider("Month ", 1, 12)

        selected_month_name = months_dict[mon]
        st.write(f"Selected Month: {selected_month_name}")
        dur = st.number_input("Call Duration (in seconds)", min_value=int(data.dur.min()), max_value=int(data.dur.max()))
        num_calls = st.number_input("Number of Calls", min_value=int(data.num_calls.min()), max_value=int(data.num_calls.max()))

        marital_options = ['Select a status'] + list(data.marital.unique())
        marital = st.selectbox("Marital Status", marital_options)

        if marital == 'married':
            marital = 0
        elif marital == 'divorced':
            marital = 1
        elif marital == 'single':
            marital = 2
           
        
        prev_outcome_options = ['Select an outcome'] + list(data.prev_outcome.unique())
        prev_outcome = st.selectbox("Previous Outcome", prev_outcome_options)

        if prev_outcome == 'unknown':
            prev_outcome = 0
        elif prev_outcome == 'failure':
            prev_outcome = 1
        elif prev_outcome == 'other':
            prev_outcome = 2
        elif prev_outcome == 'success':
            prev_outcome = 3

          

        # Assuming you have the following feature names
        feature_names = ['age', 'job', 'marital', 'education_qual', 'call_type', 'day', 'mon', 'dur', 'num_calls', 'prev_outcome']
        # Create a dictionary with feature names as keys and corresponding values
        input_data_dict = {
            'age': age,
            'day': day,
            'dur': dur,
            'mon': mon,
            'num_calls': num_calls,
            'job': job,
            'marital': marital,
            'education_qual': education_qual,
            'prev_outcome': prev_outcome,
            'call_type': call_type
            
        }

        # Create the input DataFrame with specific column names
        input = pd.DataFrame([input_data_dict], columns=feature_names)



        st.write("Input Data:")
        st.write(input)

# Make the prediction
        if st.button("Predict"):
            valu = loaded_classifier.predict(input)
            if valu == 0:
                st.write('DECLINED')
            else:
                st.write('ACCEPTED')
                st.snow()

        if st.button("About"):
            st.text("Lets Learn")
            st.text("Built with Streamlit")

if __name__=='__main__':
    main()
