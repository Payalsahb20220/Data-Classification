import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix , accuracy_score
from tabulate import tabulate, tabulate_formats
from scipy.spatial import distance

df=pd.read_csv("SteelPlateFaults-2class.csv")

x=df[df.columns[:-1]]# feature
y=df[df.columns[-1]]# label

x_train, x_test ,y_train ,y_test=train_test_split(x,y,train_size=0.7,random_state=42,shuffle=True) # splitting into tain and test
train = pd.concat([x_train, y_train], axis = 1)
test = pd.concat([x_test, y_test], axis = 1)
train.to_csv("SteelPlateFaults-train.csv") # creating csv file and save
test.to_csv("SteelPlateFaults-test.csv")

# a , b
for i in (1,3,5):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    predict=knn.predict(x_test)
    print("knn for k=", i)
    print("confusion matrix ")
    print(confusion_matrix(y_test,predict))
    print("classification accuracy ",accuracy_score(y_test,predict))

#  Q2 NORMALISATION OF TRAIN  AND TEST CSV FILE
df1=x_train
df2=x_test
min=df1.min()
max=df1.max()
diff=max-min
normalised_df1=(df1-min)/diff
normalised_df2=(df2-min)/diff
nor_train = pd.concat([normalised_df1, y_train], axis = 1)
nor_test = pd.concat([normalised_df2, y_test], axis = 1)
nor_train.to_csv("SteelPlateFaults-train-Normalised.csv")
nor_test.to_csv("SteelPlateFaults-test-normalised.csv")

for i in (1,3,5):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(normalised_df1,y_train)
    predict=knn.predict(normalised_df2)
    print(" ")
    print("knn for k=", i)
    print(" ")
    print("confusion matrix ")
    print(confusion_matrix(y_test,predict))
    print("classification accuracy ", accuracy_score(y_test,predict))

# Q3  BAYES CLASSIFIER

train=train.drop(['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'],axis=1) #train
test=test.drop(['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'],axis=1) #test
class_train=train.groupby('Class')
class0=class_train.get_group(0)
class1=class_train.get_group(1)

class0=class0.drop(['Class'],axis=1)                     
class1=class1.drop(['Class'],axis=1)

mean_class0=np.array(class0.mean())
cov_class0=np.array(class0.cov())
m0=[format(i,".3f") for i in mean_class0]
c0=np.round(cov_class0,4)

mean_class1=np.array(class1.mean())
cov_class1=np.array(class1.cov())
m1=[format(i,".3f") for i in mean_class1]
c1=np.round(cov_class1,4)
pd.options.display.width=None
print(tabulate(pd.DataFrame({"0" :m0, "1":m1})))
print("covariance matrix for class=0 \n",(c0))
print("covariance matrix for class=1 \n",(c1))

prior_class0=len(class0)/len(train)
prior_class1=len(class1)/len(train)

def likelihood(x, m, cov):                                  #likelihood function based on the bayes model
    ex = np.exp(-0.5*np.dot(np.dot((x-m).T, np.linalg.inv(cov)), (x-m)))
    return(ex/((2*np.pi)**5 * (np.linalg.det(cov))**0.5))

predict = []
x_test=x_test.drop(['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'],axis=1)
for i, x in x_test.iterrows():                              #classifying based on maximum likelihood
    p0 = likelihood(x, mean_class0, cov_class0) * prior_class0
    p1 = likelihood(x, mean_class1, cov_class1) * prior_class1
    if p0 > p1:
        predict.append(0)
    else:
        predict.append(1)

print("\nFor Bayes classifier")
print("Confusion Matrix => \n", confusion_matrix(y_test, predict))          #confusion matrix
print(" Bayes classifier Accuracy:  ", accuracy_score(y_test, predict))                #accuracy score


# Q4 comparaing best result of knn , knn normalised, bayes classifier
l=[['classification accuracy(k=5) of knn',0.8958333333333334],['classification accuracy(k=3) of normalised knn', 0.9702380952380952],['Accuracy of bayes classifier', 0.9583333333333334]]
print(tabulate(l))