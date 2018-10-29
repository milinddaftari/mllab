import numpy as np
import pandas as pd
mush = pd.read_csv("flu.csv")
mush.replace('?',np.nan,inplace=True)
print(len(mush.columns),"columns, after dropping NA,",len(mush.dropna(axis=1).columns))

#drop wherever you have ? the values are not known
mush.dropna(axis=1,inplace=True)

#the first column in dataset is class which is target variable
target = 'flu'
features = mush.columns[mush.columns != target]
classes = mush[target].unique()
test = mush.sample(frac=0.3)
mush = mush.drop(test.index)
probs = {}
probcl = {}

for x in classes:
    mushcl = mush[mush[target]==x][features]  
    clsp = {}
    tot = len(mushcl) 
    for col in mushcl.columns:
        colp = {}
        for val,cnt in mushcl[col].value_counts().iteritems():
            #df = pd.DataFrame({'mycolumn': [1,2,2,2,3,3,4]})
             #for val, cnt in df.mycolumn.value_counts().iteritems():
            # print 'value', val, 'was found', cnt, 'times'
             # value 2 was found 3 times
             #value 3 was found 2 times
              #value 4 was found 1 times
               #value 1 was found 1 times
            
            pr = cnt/tot           
            colp[val] = pr
            clsp[col] = colp            
   
    probs[x] = clsp   
    probcl[x] = len(mushcl)/len(mush)

def probabs(x):
    #X - pandas Series with index as feature
    if not isinstance(x,pd.Series):
        raise IOError("Arg must of type Series")
    probab = {}
  
    for cl in classes:
        pr = probcl[cl]
        for col,val in x.iteritems():
            try:
                pr *= probs[cl][col][val]
            except KeyError:
                pr = 0
        probab[cl] = pr
    return probab

def classify(x):   
    probab = probabs(x)  
    mx = 0
    mxcl = ''
    for cl,pr in probab.items():
        if pr > mx:
            mx = pr
            mxcl = cl
    return mxcl

#Train data
b = []
for i in mush.index:
   # print(classify(mush.loc[i,features]),mush.loc[i,target])
    b.append(classify(mush.loc[i,features]) == mush.loc[i,target])
print(sum(b),"correct of",len(mush))
print("Accuracy:", sum(b)/len(mush))
#Test data
b = []
for i in test.index:
    #print(classify(mush.loc[i,features]),mush.loc[i,target])
    b.append(classify(test.loc[i,features]) == test.loc[i,target])
print(sum(b),"correct of",len(test))
print("Accuracy:",sum(b)/len(test))