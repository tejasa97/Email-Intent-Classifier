#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import required modules
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize

#input required training and test datasets
train_data = pd.read_csv('NLP_Train_data.csv')
test_data = pd.read_csv('NLP_test_Data.csv')

#Select only required columns
train_data = train_data[['question_text', 'question_topic']]


# In[2]:


#Split into 80:20 ratio for cross validation
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(train_data['question_text'], train_data['question_topic'], test_size = 0.2, random_state = 4)


# In[3]:


#To take into consideration term frequencies of text and inverse document frequencies
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(min_df=2,stop_words='english')
X_traincv=cv.fit_transform(X_train)
X_testcv = cv.transform(X_test)
X_traincv.shape


# In[4]:


#Using MNB for classification
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_traincv,Y_train)
print(mnb.score(X_testcv, Y_test) * 100)


# In[5]:


## Accuracy of 91.5 %


# In[6]:


#Making custom query
query = "I'm considering buying either the women's detox Cleanser formula Item # 318505 or the advanced cleansing detox program Item # 707621 but wanted to ask you which is better for eliminating parasites?"
query = cv.transform([query])
mnb.predict(query)


# In[7]:


#Now to make predictions on test_data, fit TFIDV on entire training dataset 
cv1 = TfidfVectorizer(min_df=2,stop_words='english')
x_new_traincv = cv1.fit_transform(train_data['question_text'])
y_new_traincv = train_data['question_topic']

#Use the test csv as test data
x_test_new = test_data['question_text']
x_test_newcv = cv1.transform(x_test_new)


# In[8]:


#Training on entire training dataset
mnb_new = MultinomialNB()
mnb_new.fit(x_new_traincv, y_new_traincv)


# In[9]:


#Make the predictions on the test dataset
pred = mnb_new.predict(x_test_newcv)


# In[11]:


#Output the predictions to csv format in format 
ids = [i+1 for i in range(len(pred))]
test_out = pd.DataFrame({"ID" : ids, "Question Topic" : pred})
test_out.to_csv("test_out.csv", index = False)

