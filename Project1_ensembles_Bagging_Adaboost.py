import pandas as pd 
import time
from sklearn.preprocessing import MinMaxScaler 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn import model_selection 
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('D:\\Data Mining\\new_data.csv') 
feature_cols = ['gameDuration','seasonId','firstBlood','firstTower','firstInhibitor','firstBaron','firstDragon','firstRiftHerald','t1_towerKills',
           't1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills'] 
X = data[feature_cols] # Features 
scaler = MinMaxScaler(feature_range=(0, 1)) 
normalizedData = scaler.fit_transform(X) 
x = normalizedData[:,0:18] 
Y = data['winner'] 

#Bagging
kfold = model_selection.KFold(n_splits=10, random_state=1) 
cart = DecisionTreeClassifier() 
num_trees = 100 
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees,random_state=1) 

start1=time.perf_counter()
results = model_selection.cross_val_score(model, x, Y, cv=kfold) 
end1=time.perf_counter()
a1=end1-start1
print(results.mean()) 
print("running time(s) is: ",a1)
print('\n')


#Adaboost
seed = 7 
num_trees = 70 
kfold2 = model_selection.KFold(n_splits=10, random_state=seed) 
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed) 

start2=time.perf_counter()
results_2 = model_selection.cross_val_score(model, x, Y, cv=kfold2) 
end2=time.perf_counter()
a2=end2-start2
print(results_2.mean())
print("running time(s) is: ",a2)

