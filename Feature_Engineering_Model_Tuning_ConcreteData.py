#!/usr/bin/env python
# coding: utf-8

# In[2]:


### ENABLING GRAPH PLOTTING IN JUPYTER
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


### LIBRARIES
import pandas as pd
import numpy as np

from scipy.stats import zscore
from sklearn.tree import DecisionTreeRegressor


# In[4]:


### READ FILE WITH pd.read_csv
concrete_df = pd.read_csv("concrete.csv")


# In[5]:


len(concrete_df)


# In[6]:


concrete_df.describe().transpose()


# In[7]:


### DATA TYPES OF DATAFRAME
concrete_df.dtypes


# In[8]:


# FLOAT: 8
# INTEGER: 1


# In[9]:


concrete_df.describe()


# In[10]:


actual_strength = concrete_df.strength


# In[11]:


### PAIRPLOT WITH SEABORN
import seaborn as sns
sns.pairplot(concrete_df , diag_kind = 'kde')


# In[12]:


### CONVERT VALUES TO Z-SCORE
concrete_df_z = concrete_df.apply(zscore)


# In[13]:


concrete_df_z = pd.DataFrame(concrete_df_z , columns  = concrete_df.columns)
concrete_df_z.describe()


# In[14]:


y = concrete_df_z[['strength']]
X = concrete_df_z.drop(labels= "strength" , axis = 1)


# In[15]:


### CREATE TRAIN AND TEST DATA
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)


# In[16]:


#### DECISION TREE ####


# In[17]:


### MODEL
dt_model = DecisionTreeRegressor()


# In[18]:


### FIT THE MODEL TO ONLY THE TRAIN DATA
dt_model.fit(X_train, y_train)


# In[19]:


print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))


# In[20]:


### SCORE THE TEST DATA
dt_model.score(X_test, y_test)


# In[21]:


### SCORE THE TRAIN DATA
dt_model.score(X_train, y_train)


# In[22]:


### DROP COLUMNS THAT ARE NOT NEEDED
# *ASH
# *COARSEAGG
# *FINEAGG
# *SUPERPLASTIC
# *STRENGTH

drop_cols = ['ash' , 'coarseagg' , 'fineagg' , 'superplastic' , 'strength']

X = concrete_df_z.drop(labels= drop_cols , axis = 1)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30, random_state=1)


# In[24]:


dt_model = DecisionTreeRegressor(min_samples_leaf=20)


# In[25]:


dt_model.fit(X_train, y_train)


# In[26]:


print (pd.DataFrame(dt_model.feature_importances_, columns = ["Imp"], index = X_train.columns))


# In[27]:


dt_model.score(X_test, y_test)


# In[28]:


dt_model.score(X_train, y_train)


# In[29]:


#### VISUALIZE DECISION TREE ####


# In[30]:


### LIBRARIES
from IPython.display import Image  
from sklearn import tree
from os import system


# In[31]:


Credit_Tree_File = open('d:\concrete_tree.dot','w')
dot_data = tree.export_graphviz(dt_model, out_file=Credit_Tree_File, feature_names = list(X_train))

Credit_Tree_File.close()


# In[32]:


#### K-MEANS ####


# In[33]:


### LIBRARY
from sklearn.cluster import KMeans


# In[34]:


cluster_range = range( 1, 15 )
cluster_errors = []
for num_clusters in cluster_range:
  clusters = KMeans( num_clusters, n_init = 10 )
  clusters.fit(concrete_df_z)
  labels = clusters.labels_
  centroids = clusters.cluster_centers_
  cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
clusters_df[0:15]


# In[35]:


### ELBOW PLOT
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )


# In[36]:


kmeans = KMeans(n_clusters= 6)
kmeans.fit(concrete_df_z)


# In[37]:


labels = kmeans.labels_
counts = np.bincount(labels[labels>=0])
print(counts)


# In[38]:


### NEW DATAFRAME FOR LABELS converting 
### CONVERTNG LABELS TO CATEGORICAL VARIABLE
cluster_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))
cluster_labels['labels'] = cluster_labels['labels'].astype('category')
concrete_df_labeled = concrete_df.join(cluster_labels)

concrete_df_labeled.boxplot(by = 'labels',  layout=(3,3), figsize=(30, 20))


# In[39]:


# NO VISIBLE DISTINCT CLUSTERS
# CEMENT ONLY STRONG PREDICTOR
# CONCLUSTION: CLUSTERS WILL NOT YIELD DESIRED RESULT


# In[40]:


#### ENSEMBLE TECHNIQUES ####


# In[41]:


### LIBRARIES
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn import model_selection
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn import metrics


# In[42]:


### GRADIENT BOOST ###

gbmTree = GradientBoostingRegressor(n_estimators=50)
gbmTree.fit(X_train,y_train)
print("gbmTree on training" , gbmTree.score(X_train, y_train))
print("gbmTree on test data ",gbmTree.score(X_test,y_test))


# In[43]:


### BAGGING ###

bgcl = BaggingRegressor(n_estimators=100, oob_score= True)
bgcl = bgcl.fit(X_train,y_train)
print("bgcl on train data ", bgcl.score(X_train,y_train))
print("bgcl on test data ", bgcl.score(X_test,y_test))


# In[44]:


### RANDOM FOREST ###

rfTree = RandomForestRegressor(n_estimators=100)
rfTree.fit(X_train,y_train)
print("rfTree on train data ", rfTree.score(X_train,y_train))
print("rfTree on test data ", rfTree.score(X_test,y_test))


# In[45]:


concrete_XY = X.join(y)


# In[46]:


### BOOTSTRAP ###

# * BOOTSTRAP SAMPLES: 1000

values = concrete_XY.values

n_iterations = 1000       
n_size = int(len(concrete_df_z) * 1)    


stats = list()  
for i in range(n_iterations):
    train = resample(values, n_samples=n_size)  
    test = np.array([x for x in values if x.tolist() not in train.tolist()])  
    
    
    # FIT MODEL
    gbmTree = GradientBoostingRegressor(n_estimators=50)
    gbmTree.fit(train[:,:-1], train[:,-1])   
    y_test = test[:,-1]    

    # MODEL EVALUATION & PREDICTION
    predictions = gbmTree.predict(test[:, :-1])   
    score = gbmTree.score(test[:, :-1] , y_test)

    stats.append(score)


# In[47]:


### PLOT THE SCORE

from matplotlib import pyplot
pyplot.hist(stats)
pyplot.show()

# 95% CONFIDENCE
alpha = 0.95                             
p = ((1.0-alpha)/2.0) * 100              
lower = max(0.0, np.percentile(stats, p))  
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


# In[ ]:


n_iterations = 1000              
n_size = int(len(concrete_df_z) * 1)  

stats = list()   

for i in range(n_iterations):


    train = resample(values, n_samples=n_size)  
    test = np.array([x for x in values if x.tolist() not in train.tolist()])  


    rfTree = RandomForestRegressor(n_estimators=100)  
    rfTree.fit(train[:,:-1], train[:,-1])   
    
    rfTree.fit(train[:,:-1], train[:,-1])   
    y_test = test[:,-1]    

    predictions = rfTree.predict(test[:, :-1])   
    score = rfTree.score(test[:, :-1] , y_test)

    stats.append(score)


# In[ ]:


from matplotlib import pyplot
pyplot.hist(stats)
pyplot.show()

alpha = 0.95                             
p = ((1.0-alpha)/2.0) * 100              
lower = max(0.0, np.percentile(stats, p))  
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(stats, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))


# In[ ]:


#### MODEL TUNING ####


# In[ ]:


### USE HYPERPARAMETERS


# In[ ]:


from pprint import pprint


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rf = RandomForestRegressor(random_state = 1)


# In[ ]:


print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[ ]:


#### RANDOM SEARCH CV ####


# In[ ]:


import numpy as np
print(np.linspace(start = 5, stop = 10, num = 2))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


### RANDOM FOREST NUMBER OF TREES
n_estimators = [int(x) for x in np.linspace(start = 10 , stop = 15, num = 2)]   

### NUMBER OF FEATURES
max_features = ['auto', 'sqrt']

### MAXIMUM NUMBER LEVELS IN TREE
max_depth = [int(x) for x in np.linspace(5, 10, num = 2)]   
max_depth.append(None)

# MINIMUM SAMPLE NUMBERS TO SPLIT NODE
min_samples_split = [2, 5, 10]

# MINIMUM SAMPLE NUMBERS AT EACH LEAF
min_samples_leaf = [1, 2, 4]

# SAMPLE SELECTION METHOD FOR TRAINING TREES
bootstrap = [True, False]

# RANDOM GRID
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


# In[ ]:


### RANDOM GRID SEARCH FOR BEST HYPERPARAMETERS
### 3 FOLD VALIDATION
### 100 DIFFERENT COMBINATIONS
### USE ALL AVAILABLE CORES

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 5, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)

### FIT RANDOM SEARCH MODEL
rf_random.fit(X_train, y_train);


# In[ ]:


rf_random.best_params_


# In[ ]:


best_random = rf_random.best_estimator_

best_random.score(X_test , y_test)


# In[ ]:


#### GRID SEARCH CV ####


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


param_grid = {
    'bootstrap': [True],
    'max_depth': [5,6],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4],
    'min_samples_split': [5,10],
    'n_estimators': [5,6,7]
}  



rf = RandomForestRegressor(random_state = 1)



grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = 1, verbose = 0, return_train_score=True)



### GRID SEARCH FIT
grid_search.fit(X_train, y_train);


# In[ ]:


grid_search.best_params_


# In[ ]:


best_grid = grid_search.best_estimator_
best_grid.score(X_test, y_test)


# In[ ]:


#### END ####

