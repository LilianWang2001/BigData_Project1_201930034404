# Load libraries 
import pandas as pd 
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score #for accuracy calculation 
from sklearn.metrics import precision_score #for precision calculation
from sklearn.metrics import recall_score #for recall calculation
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier 
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import svm
from sklearn.neural_network import MLPClassifier



col_names=['gameId','creationTime','gameDuration','seasonId','winner','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills',
           't1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
# load dataset 
lol = pd.read_csv("D:\\Data Mining\\new_data.csv", header=None, names=col_names) # Pay attention to the address where the data is stored 
lol = lol.iloc[1:] # delete the first row of the dataframe 
lol.head() 


TestSet= pd.read_csv("D:\\Data Mining\\test_set.csv", header=None, names=col_names) # Pay attention to the address where the data is stored 
TestSet = TestSet.iloc[1:] # delete the first row of the dataframe 
TestSet.head() 



#split dataset in features and target variable 
feature_cols = ['gameDuration','seasonId','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills',
           't1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
X = lol[feature_cols] # Features 
y = lol.winner # Target variable 
#normalized data
scaler = MinMaxScaler(feature_range=(0, 1)) 
X = scaler.fit_transform(X) 




x_test=TestSet[feature_cols]
Y_test=TestSet.winner
x_test = scaler.fit_transform(x_test)

# Split dataset into training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test



start1=time.perf_counter()
# Classifier 1 DecisionTreeClassifier 
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5) 

#  the accuracy when using "gini" is higher than using "entropy"


# Train Decision Tree Classifer 
clf = clf.fit(X_train,y_train) 
end1=time.perf_counter()
#Predict the response for test dataset 
y_pred = clf.predict(X_test) 
 
test_pred = clf.predict(x_test) 


# Model Accuracy, how often is the classifier correct? 
print("Decision Tree Accuracy:",accuracy_score(y_test, y_pred)) 
print("Decision Tree Test Accuracy:",accuracy_score(Y_test, test_pred)) 
print("Decision Tree Precision:",precision_score(Y_test, test_pred,average='micro'))
print("Decision Tree Recall:",recall_score(Y_test, test_pred,average='micro'))
a1=end1-start1
print("Decision Tree running time(s) is: ",a1)
print('\n')

from sklearn.tree import export_graphviz 
from six import StringIO   
from IPython.display import Image   
import pydotplus 
import os      
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/' 
# Configure environment variables 
dot_data = StringIO() 
export_graphviz(clf, out_file=dot_data,   
                filled=True, rounded=True, 
                special_characters=True,feature_names = feature_cols,class_names=['0','1']) 
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())   
graph.write_png('lol.png') 
Image(graph.create_png()) 

import matplotlib.pyplot as plt # Plt is used to display pictures 
import matplotlib.image as mpimg # Mpimg is used to read pictures 
lol = mpimg.imread('lol.png') # Read diabetes.png in the same directory as the code 
# Diabetes is already an np.array and can be processed at will 
plt.imshow(lol) # Show Picture 
plt.axis('off') # Do not show axes 
plt.show() 
 








#Classifier 2  SVM

cls_svm=svm.SVC(kernel='rbf',gamma=0.015,C=20)
#cls_svm=svm.SVC()
start2=time.perf_counter()
cls_svm.fit(X_train,y_train)
end2=time.perf_counter()
#Predict the response for test dataset 
y_svm_pred = cls_svm.predict(X_test) 
 
test_svm_pred = cls_svm.predict(x_test) 

# Model Accuracy, how often is the classifier correct? 
print("SVM Accuracy:",accuracy_score(y_test, y_svm_pred)) 
print("SVM Test Accuracy:",accuracy_score(Y_test, test_svm_pred)) 
print("SVM Precision:",precision_score(Y_test, test_svm_pred,average='micro'))
print("SVM Recall:",recall_score(Y_test, test_svm_pred,average='micro'))
a2=end2-start2
print("SVM running time(s) is: ",a2)
print('\n')



#Classifier 3 MLP

cls_mlp=MLPClassifier(alpha=0.0001,learning_rate='constant',learning_rate_init=0.01,random_state=1)
#cls_mlp=MLPClassifier(random_state=1)
start3=time.perf_counter()
cls_mlp.fit(X_train,y_train)
end3=time.perf_counter()

y_mlp_pred = cls_mlp.predict(X_test) 
 
test_mlp_pred = cls_mlp.predict(x_test) 

# Model Accuracy, how often is the classifier correct? 
print("MLP Accuracy:",accuracy_score(y_test, y_mlp_pred)) 
print("MLP Test Accuracy:",accuracy_score(Y_test, test_mlp_pred))
print("MLP Precision:",precision_score(Y_test, test_mlp_pred,average='micro'))
print("MLP Recall:",recall_score(Y_test, test_mlp_pred,average='micro'))
a3=end3-start3
print("MLP running time(s) is: ",a3)


