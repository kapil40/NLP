import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
#Used to import the dataset
from sklearn.datasets import load_files
nltk.download('stopwords')


#Importing dataset

reviews=load_files('txt_sentoken/')
X,y=reviews.data,reviews.target   #Use to seperate the documents and it's target class


#Storing as pickle files
#w for write and b for byte as in pickle format data is stored in form of byte
with open('X.pickle','wb') as f:
    pickle.dump(X,f) 

with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
#Unpickling the dataset 
#Use to retrieve the data, it is very helping for persisting the data and helps to improve the performance significantly

with open('X.pickle','rb') as f:
    X=pickle.load(f)

with open('y.pickle','rb') as f:
    y=pickle.load(f)
    
#Creating the corpus

corpus=[]       #It contains list of preprocessed documents
for i in range(0,len(X)):
    review=re.sub(r'\W',' ',str(X[i]))       #Used to remove the punctuations marks, ascii value, semicolon
    review=review.lower()                    #Convert it into lower case
    review=re.sub(r'\s+[a-z]\s+',' ',review) #Used to remove the single characters
    review=re.sub(r'^[a-z]\s+',' ',review)   #Remove the single character from the start of the sentence
    review=re.sub(r'\s+',' ',review)         #Remove the extra spaces genereted 
    corpus.append(review)                    #Append preprocessed review into corpus
 
#Creating the BOW model
#CountVectorizer is used to create the BOW model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=3000,min_df=3,max_df=0.6,stop_words=stopwords.words('english'))
#max_features is used to take top n frequent words
#min_df is minimum document frequency is used to exclude a word if it occurs less than n times
#max_df is maximum document frequency is used to exclude a word if it occurs more than n percent
X=vectorizer.fit_transform(corpus).toarray()    
#Store the BOW model in X and get the n dimensional array of BOW model

#Transform BOW model to Tfidf model
from sklearn.feature_extraction.text import TfidfTransformer
transformer=TfidfTransformer()
X=transformer.fit_transform(X).toarray()  #It contains floating point numbers

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(max_features=3000,min_df=5,max_df=0.6,stop_words=stopwords.words('english'))
X=vectorizer.fit_transform(corpus).toarray()

#Used to seperate the training and testing dataset 
from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test=train_test_split(X,y,test_size=0.2,random_state=0) #80% for training and 20% for testing
#text_train comprises of list of documents which will be used to train the model
#text_test comprises of list of documents which will be used to test the model
#sent_train are the sentiments associated with text_train
#sent_test are the sentiments associated with text_test

#Applying the Logistic Regression to the model fo classifying
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(text_train,sent_train)

#Predicting the result
sent_pred=classifier.predict(text_test)

#Creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(sent_test,sent_pred)

#Pickling the classifier as it will be easy to classify, it is a pretrained model 

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)


#Pickling the vectorizer , it is a pretrained model

with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)

#Unpickling classifier and vectorizer and storing it

with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)          #It is imported classifier

with open('tfidfmodel.pickle','rb') as f:
     tfidf=pickle.load(f)
    
sample=input("enter comment:")
sample=sample.split(",")
    
sample=tfidf.transform(sample).toarray()        #transform on based on current corpus, the document will look like corpus
#print(clf.predict(sample))
if(clf.predict(sample)==1):
    print("Positive Comment")
else:
    print("Negative Comment")

