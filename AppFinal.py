#!/usr/bin/env python
# coding: utf-8

# #### Q1 - Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[1]:


import pandas as pd
s = pd.read_csv('social_media_usage.csv')



# #### Q2 - Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[2]:


import numpy as np
import pandas as pd
def clean_sm(x):
    x = np.where(x==1, 1, 0)
    return(x)
clean_sm(1)


# #### Q3 - Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[3]:


ss = pd.DataFrame({   
    "sm_li":np.where(s["web1h"] ==1 , 1, 0),   
    "income":np.where(s["income"] > 9, np.nan,s["income"]),   
    "education":np.where(s["educ2"] >8, np.nan,s["educ2"]),   
    "parent":np.where(s["par"] ==1,1,0),  
    "marital":np.where(s["marital"] ==1,1,0),   
    "female":np.where(s["gender"] ==2,1,0), 
    "age":np.where(s["age"] >98,np.nan,s["age"]) 
}).dropna() 
print(ss)


# In[4]:


y=ss["sm_li"]
X=ss[["age","female","marital","parent","education","income"]]
print(y,X)


# #### Q5 - Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[5]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    stratify=y,       # same number of target in training & test set 
                                                    test_size=0.2,    # hold out 20% of data for testing 
                                                    random_state=987) # set for reproducibility 
# X_train contains 80% of the data and contains the features used to predict the target when training the model.  
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance.  
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model.  
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.


# #### Q6 - Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[6]:


from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(class_weight='balanced', random_state=0) 
lr.fit(X_train,y_train)


# #### Q7 - Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[7]:

 
from sklearn.metrics import classification_report  
from sklearn.metrics import accuracy_score

# Make predictions using the model and the testing data 
y_pred = lr.predict(X_test) 
########Accuracy
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred)) 
##My Created Confussion Matrix of model 
import matplotlib.pyplot as plt 
import pylab as pl 

# In[9]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score 
print('Precision: %.3f' % precision_score(y_test, y_pred)) 
print('Recall: %.3f' % recall_score(y_test, y_pred)) 
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred)) 
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #### Q10 - Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[24]:


import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
from PIL import Image
image = Image.open('Georgetown.jpg')
st.image(image, caption='Georgetown University MSBA Project')
st.header("Predicting if you are a LinkedIn user")

user_age = st.number_input('Please Enter or Toggle to your age', min_value=10,max_value=97,value=18, step=1)
st.write('Your Chosen Age Is ', user_age)
user_gender=st.selectbox("Please provide your gender? If Female chose ONE, if Male chose ZERO", options=[1,0])
user_married=st.selectbox("Please provide your Marital Status? If Married chose ONE, if not chose ZERO", options=[1,0])
user_parent=st.selectbox("Are you a parent? If so chose ONE, if not chose ZERO", options=[1,0])
st.markdown("What is your income - 1= Less $10k 2= $10k-$20k 3= $20k-$30k 4=$30k-$40k 5=$40k-$50k 6=$50k-$75k 7= $75-$100 8= $100-$150")
user_income=st.slider("What is your income", min_value=1, max_value=8,value=1,step=1)
st.markdown("1=Less than high school,2=High school incomplete, 3=High school graduate,4=Some college, no degree,5=Two-year associate degree,6=Four-year college or university degree,7=Some postgraduate or professional schooling")
user_education=st.slider("What is your education",min_value=1,max_value=7,value=1,step=1)

person=[user_age,user_gender,user_married,user_parent,user_income,user_education]
predicted_class =lr.predict([person])
probs=lr.predict_proba([person])

st.subheader('Based on provided information, see below to learn your likeness to use Linkedin below')
if predicted_class >0:
    label="You are a Linkedin user!"
else:
    label="You are not a Linkedin user"
st.write(label)

st.subheader("Your probability to utilizate Linkedin")
st.write(probs)


st.subheader("Learn More About Georgetowns MBA Programs")
video_file = open('Your Moment of Truth.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)


# In[ ]:
