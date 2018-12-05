import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

msg = pd.read_csv('naivetrext1.csv',names=['message','label'])
print('\n DATASET DIMENSIONS : ',msg.shape)
msg['labelnum'] = msg.label.map({'pos':1,'neg':0})
X = msg.message
y = msg.labelnum
xtrain,xtest,ytrain,ytest=train_test_split(X,y)
cv = CountVectorizer()
xtrain_dtm = cv.fit_transform(xtrain)
xtest_dtm = cv.transform(xtest)
print(cv.get_feature_names())
df = pd.DataFrame(xtrain_dtm.toarray(),columns=cv.get_feature_names())
print(df)
clf = MultinomialNB().fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)
print('Accuracy metrics')
print('Accuracy of the classifer is',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(ytest,predicted))
print('Recall and Precison ')
print(metrics.recall_score(ytest,predicted))
print(metrics.precision_score(ytest,predicted))