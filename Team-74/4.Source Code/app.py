import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np

model1 = joblib.load('model_pp1.pkl')
model2 = joblib.load('model_pp2.pkl')
loaded_vectorizer = joblib.load("vectorizer.joblib")

st.title("Welcome to Placement Prediction")

Name = st.text_input("Name:")
a=st.radio(
    "Gender(Male[1]/Female[0]:",
    (1,0))
b=st.radio(
    "Stream(Other[0]/ECE[1]/CSE[2]):",
    (0,1,2))
c = int(st.number_input("Technical Skill Percentage:", min_value=0, max_value=100, step=1))
d=st.radio(
    "Self-Learning Capability(yes[1]/No[0]):",
    (1,0))
e = int(st.number_input("Coding_perc:", min_value=0, max_value=100, step=1))
f=st.radio(
    "Extra Courses(yes[1]/No[0]):",
    (1,0))
g = int(st.number_input("Communication_perc:", min_value=0, max_value=100, step=1))
h=st.radio(
    "Internship(yes[1]/No[0]):",
    (1,0))
i = int(st.number_input("Logical Reasoning perc:", min_value=0, max_value=100, step=1))

o1=['shell programming','machine learning','app development','python',
 'r programming','information security','hadoop','distro making',
 'full stack']
j1 = st.selectbox(
         'Certifications',o1)
o2 = ['cloud computing','database security','web technologies','data science',
 'testing','hacking','game development','system designing']
j2 = st.selectbox(
         'Workshops',o2)
o3 = ['cloud computing','networks','hacking','Computer Architecture',
 'programming','parallel computing','IOT','data engineering',
 'Software Engineering','Management']

j3 = st.selectbox(
         'Interested Subjects',o3)





btn=st.button("predict")
new_data=np.array([a,b,c,d,e,f,g,h,i]).reshape(1,-1)


if btn:
    pred=model1.predict(new_data)
    prob=model1.predict_proba(new_data)

    if pred==1:
        st.write('Placed')
        st.write(f"You will be placed with probability of {prob[0][1]:.2f}")
        new_job_description = f"{j1.title()} {j2.title()} {j3.title()}"
        new_job_vector = loaded_vectorizer.transform([new_job_description])
        predicted_job_role = model2.predict(new_job_vector)
        st.write("Suggested Job Role:", predicted_job_role)
    else:
        st.write('Not-Placed')
        st.write("You need to improve your Skills !")
        st.write("Here are some websites for your Reference...")
        st.write("https://www.geeksforgeeks.org/")
        st.write("https://www.javatpoint.com/")


   
