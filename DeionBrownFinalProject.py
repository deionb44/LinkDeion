#!/usr/bin/env python
# coding: utf-8

#  # Final Project
# ## Deion Brown
# ### 12/5/2022 

# *** 

# #### Q1 - Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[52]:


import os
os.chdir('C:/Users/toytr/OneDrive/GTOWN/OPIM 607/Final Project')
print(os.getcwd())
import pandas as pd
s = pd.read_csv('social_media_usage.csv')
s.shape


# ***

# #### Q2 - Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[66]:


import numpy as np
import pandas as pd
def clean_sm(x):
    x = np.where(x==1, 1, 0)
    return(x)
clean_sm(1)


# #### Q3 - Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[71]:


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


# ***

# #### Q4 -Create a target vector (y) and feature set (X)

# In[88]:


y=ss["sm_li"]
X=ss[["age","female","marital","parent","education","income"]]
print(y,X)


# ***

# #### Q5 - Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[87]:


from sklearn.model_selection import train_test_split 
# Split data into training and test set 
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    stratify=y,       # same number of target in training & test set 
                                                    test_size=0.2,    # hold out 20% of data for testing 
                                                    random_state=987) # set for reproducibility 
# X_train contains 80% of the data and contains the features used to predict the target when training the model.  
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance.  
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model.  
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.


# ***

# #### Q6 - Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[89]:


from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(class_weight='balanced', random_state=0) 
lr.fit(X_train,y_train)


# ***

# #### Q7 - Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[104]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import plot_confusion_matrix 
# Make predictions using the model and the testing data 
y_pred = lr.predict(X_test) 
########Accuracy
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred)) 
##My Created Confussion Matrix of model 
import matplotlib.pyplot as plt 
import pylab as pl 
con = confusion_matrix(y_test, y_pred) 
pl.matshow(con) 
pl.title('Confusion matrix of the classifier') 
pl.colorbar() 
pl.show() 
###Alternative Plot 
import seaborn as sns 
import matplotlib.pyplot as plt  
ax= plt.subplot() 
sns.heatmap(con, annot=True, fmt='g', ax=ax);


# ***

# #### Q8 - Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[103]:


# Compare those predictions to the actual test data using a confusion matrix (positive class=1) 
#confusion_matrix(y_test, y_pred) 
pd.DataFrame(confusion_matrix(y_test, y_pred), 
            columns=["Predicted negative", "Predicted positive"], 
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG") 


# ***

# #### Q9 - Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# In[106]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score 
print('Precision: %.3f' % precision_score(y_test, y_pred)) 
print('Recall: %.3f' % recall_score(y_test, y_pred)) 
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred)) 
print('F1 Score: %.3f' % f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# ***

# #### Q10 - Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[151]:


import streamlit as st
from textblob import TextBlob
import plotly.graph_objects as go
st.header("Predicting if you are a LinkedIn user")

user_age=st.slider("What is your age", min_value=10,max_value=97,value=18, step=1)
user_gender=st.selectbox("what is your gender? If female select 1, if male select 0", options=[1,0])
user_married=st.selectbox("Are you a married? If so select 1, if not select 0", options=[1,0])
user_parent=st.selectbox("Are you a parent? If so select 1, if not select 0", options=[1,0])
st.markdown("What is your income - 1= Less $10k 2= $10k-$20k 3= $20k-$30k 4=$30k-$40k 5=$40k-$50k 6=$50k-$75k 7= $75-$100 8= $100-$150")
user_income=st.slider("What is your income", min_value=1, max_value=8,value=1,step=1)
st.markdown("1=Less than high school,2=High school incomplete, 3=High school graduate,4=Some college, no degree,5=Two-year associate degree,6=Four-year college or university degree,7=Some postgraduate or professional schooling")
user_education=st.slider("What is your education",min_value=1,max_value=7,value=1,step=1)

person=[user_age,user_gender,user_married,user_parent,user_income,user_education]
predicted_class =lr.predict([person])
probs=lr.predict_proba([person])

st.subheader('is this person a Linkedin user')
if predicted_class >0:
    label="this is a Linkedin user!"
else:
    label="This is not a Linkedin user"
st.write(label)

st.subheader("Whart is the probability that this person is a Linkedin user")
st.wrote(probs)


# In[153]:


person=[age,gender,married,parent,income,education]
predicted_class =lr.predict([person])
probs=lr.predict_proba([person])


# ***
