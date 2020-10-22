#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
import pickle
import numpy
from sklearn.linear_model import SGDRegressor

# In[3]:

st.cache()
def get_data():
    url = "Admission_Predict_Ver1.1.csv"
    return pd.read_csv(url,index_col='Serial No.')
data = get_data()


# In[4]:



# In[6]:


st.title("Graduate Admission Probability")
st.markdown("""This web app is based on the Graduate Admission 2 set available on [`Kaggle`](https://www.kaggle.com/mohansacharya/graduate-admissions).
            The original Notebook can be viewed from [`here`](https://www.kaggle.com/aakashg1999/graduate-admission-eda-complete). """)


# In[7]:


st.header("Credits")
st.markdown("""The dataset below is provided by Kaggle user [`Mohan S Acharya`](https://www.kaggle.com/mohansacharya/).
            We would like to thank him for providing the dataset.
             PS : The data can be sorted according to a column by selecting that column.""")


# In[ 8]:st.markdown(data.columns.tolist())
cols = data.columns.tolist()
st_ms = st.multiselect("Columns", data.columns.tolist(), default=cols)
st.dataframe(data[st_ms].head(20))

# In[Final}
st.title("Enter Data for making a prediction")
pickle_Filename='Pickle_SGD_Model.pkl'

with open(pickle_Filename, 'rb') as file:  
    Pickled_Model = pickle.load(file)

dict_std={'GRE Score':11.457694,'TOEFL Score':6.160395,'University Rating':1.152560,
          'SOP':1.005567,'LOR ':0.935844,'CGPA':0.604475,
          'Research':0.498087}
dict_mean={'GRE Score':316.130952,'TOEFL Score':107.252381,'University Rating':3.102381,
          'SOP':3.370238,'LOR ':3.490476,'CGPA':8.568738,
          'Research':0.550000}
ser_mean=pd.Series(dict_mean)
ser_std=pd.Series(dict_std)
GRE=st.number_input("Enter GRE Score",260,340,320)
TOEFL=st.number_input("Enter TOEFL Score",0,120,110)
SOP=st.number_input("Give your SOP a score from 1 to 5(5 being the best)",1.,5.,step=0.5)
LOR=st.number_input("Give your LOR a score from 1 to 5(5 being the best)",1.,5.,step=0.5)
CGPA=st.number_input("Enter your graduation CGPA",4.,10.,9.)
Research=st.number_input("Give your Research  a score from 1 to 5(5 being the best)",1.,5.,step=0.5)
Research=st.selectbox('Do you have Research Experience',
                    ('Yes','No'))
Research_bool=1
if Research[0] is 'Y':
    Research_bool=1
else:
    Research_bool=0

dict_cols={'GRE Score':[GRE,GRE,GRE,GRE,GRE],'TOEFL Score':[TOEFL,TOEFL,TOEFL,TOEFL,TOEFL],'University Rating':[1,2,3,4,5],
          'SOP':[SOP,SOP,SOP,SOP,SOP],'LOR ':[LOR,LOR,LOR,LOR,LOR],'CGPA':[CGPA,CGPA,CGPA,CGPA,CGPA],
          'Research':[Research_bool,Research_bool,Research_bool,Research_bool,Research_bool]}
df=pd.DataFrame(dict_cols)
df=(df-ser_mean)/ser_std
result=Pickled_Model.predict(df)

st.header("Prediction for University Rating 1 to 5")
st.write(result)



# In[17]
st.title("Interesting Observation")
st.markdown("It is well established that good score in all fronts increases the chance of getting an admit in universities.")
val= st.selectbox("University Rating",(1,2,3,4,5))

data.rename(columns={'University Rating':'Rating'},inplace=True)

data.loc[data.Rating==val]

st.markdown("But, as we can see, University Rating=1 has lesser chance of admit and the reason for the same is as follows :")
st.markdown("As we can see, the dataset for Rating=1 only has candidates that have less chances of admit(low scores in all fronts) thus all the data points for Rating=1 convey that chance of admit are less and so on till Rating =3 and vice versa is true for Rating=4 and 5")






