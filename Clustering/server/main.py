#!/usr/bin/env python
# coding: utf-8


# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
#import sklearn as sl
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
#from Preprocessing import Preprocessing

# %matplotlib inline



os.listdir()
#import data
df_ex_preprocessing = pd.read_csv('data/googleplaystore.csv')




#Drop unwanted columns
df = df_ex_preprocessing.drop(["App","Current Ver"],1)


#delete Type,Content Rating
df_train = df.copy()
for i in ['Type','Content Rating','Android Ver']:
    df_train = df_train.drop(df_train.loc[df_train[i].isnull()].index,0)
df_train.info()



#fill rating null values with mean quartiles
x = sum(df_train.Rating.describe()[4:8])/4
df_train.Rating = df_train.Rating.fillna(x)
#print("Dataset contains ",df_train.isna().any().sum()," Nan values.")


from sklearn.preprocessing import OneHotEncoder



#Hot encode the 'Category' and 'Genre' columns
encode = OneHotEncoder()

data_encoded = encode.fit(df_train[['Category','Genres']]).transform(df_train[['Category','Genres']])
temp = pd.DataFrame(data_encoded.toarray(),columns=encode.get_feature_names())

df_train = pd.concat([temp, df_train[['Rating', 'Reviews', 'Size', 'Installs', 'Type',
                                    'Price', 'Content Rating', 'Last Updated', 'Android Ver']]], axis = 1, join = 'inner')





#Map the content rating column
df_train['Content Rating'] = df_train['Content Rating'].map({'Unrated':0.0,
                                                 'Everyone':1.0,
                                                 'Everyone 10+':2.0,
                                                 'Teen':3.0,
                                                 'Adults only 18+':4.0,
                                                 'Mature 17+':5.0})
df_train['Content Rating'] = df_train['Content Rating'].astype(float)



#change type to float for Review
df_train['Reviews'] = df_train['Reviews'].astype(float)


#clean 'M','k', fill 'Varies with device' with median and transform to float 
lists = []
for i in df_train["Size"]:
    if 'M' in i:
        i = float(i.replace('M',''))
        i = i*1000000
        lists.append(i)
    elif 'k' in i:
        i = float(i.replace('k',''))
        i = i*1000
        lists.append(i)
    else:
        lists.append("Unknown")
    
k = pd.Series(lists)
median = k[k!="Unknown"].median()
k = [median if i=="Unknown" else i for i in k]
df_train["Size"] = k

del k,median,lists





#clean '$' in Price and transform to float 
df_train['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in df_train['Price'] ]



#Remove '+' and ',' from installs
df_train["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in df_train["Installs"] ]


#Map the 'Type' column 
df_train.Type = df_train.Type.map({'Free':0,"Paid":1})


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

#transform ' Android Ver ' column
df_train['Android Ver']=le.fit(df_train['Android Ver']).transform(df_train['Android Ver'])





from datetime import datetime


df_train["Last Updated"] = [datetime.strptime(i, '%B %d, %Y') for i in df_train["Last Updated"]]


df_train["Last Updated"] = [i.year for i in df_train["Last Updated"]]




#transform ' Last Updated ' column
df_train['Last Updated']=le.fit(df_train['Last Updated']).transform(df_train['Last Updated'])


df_train['Last Updated'].unique()



df_train.isna().any().sum()



#Normalizing the column values
from sklearn import preprocessing

normalized_X = preprocessing.normalize(df_train)
Df_final=pd.DataFrame(normalized_X)



#DUNN Index Calculator
# -*- coding: utf-8 -*-
__author__ = "Joaquim Viegas"

""" JQM_CV - Python implementations of Dunn and Davis Bouldin clustering validity indices
dunn(k_list):
    Slow implementation of Dunn index that depends on numpy
    -- basec.pyx Cython implementation is much faster but flower than dunn_fast()
dunn_fast(points, labels):
    Fast implementation of Dunn index that depends on numpy and sklearn.pairwise
    -- No Cython implementation
davisbouldin(k_list, k_centers):
    Implementation of Davis Boulding index that depends on numpy
    -- basec.pyx Cython implementation is much faster
"""

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
    
def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di

def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)
    
def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    #values = values[np.nonzero(values)]
            
    return np.max(values)

def dunn_fast(points, labels):
    """ Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    
    Parameters
    ----------
    points : np.array
        np.array([N, p]) of all points
    labels: np.array
        np.array([N]) labels of all points
    """
    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))
    
    deltas = np.ones([len(ks), len(ks)])*1000000
    big_deltas = np.zeros([len(ks), 1])
    
    l_range = list(range(0, len(ks)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)
        
        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas)/np.max(big_deltas)
    return di
    
    
def  big_s(x, center):
    len_x = len(x)
    total = 0
        
    for i in range(len_x):
        total += np.linalg.norm(x[i]-center)    
    
    return total/len_x

def davisbouldin(k_list, k_centers):
    """ Davis Bouldin Index
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    k_centers : np.array
        The array of the cluster centers (prototypes) of type np.array([K, p])
    """
    len_k_list = len(k_list)
    big_ss = np.zeros([len_k_list], dtype=np.float64)
    d_eucs = np.zeros([len_k_list, len_k_list], dtype=np.float64)
    db = 0    

    for k in range(len_k_list):
        big_ss[k] = big_s(k_list[k], k_centers[k])

    for k in range(len_k_list):
        for l in range(0, len_k_list):
            d_eucs[k, l] = np.linalg.norm(k_centers[k]-k_centers[l])

    for k in range(len_k_list):
        values = np.zeros([len_k_list-1], dtype=np.float64)
        for l in range(0, k):
            values[l] = (big_ss[k] + big_ss[l])/d_eucs[k, l]
        for l in range(k+1, len_k_list):
            values[l-1] = (big_ss[k] + big_ss[l])/d_eucs[k, l]

        db += np.max(values)
    res = db/len_k_list
    return res




#K-Means
from sklearn import cluster
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(Df_final) #K-means training
y_pred = k_means.predict(Df_final)

#print(y_pred)
# We store the K-means results in a dataframe
pred = pd.DataFrame(y_pred)
pred.columns = ['Type']

#print(pred)
# we merge this dataframe with df
prediction = pd.concat([Df_final, pred], axis = 1)
#print(prediction)

# We store the clusters
clus0 = prediction.loc[prediction.Type == 0]
clus1 = prediction.loc[prediction.Type == 1]
#clus2 = prediction.loc[prediction.Type == 2]
cluster_list = [clus0.values, clus1.values] 
#print(dunn(cluster_list))
  




#print(cluster_list)




sse = [] 
k_rng = range(1,10) 
for k in k_rng: 
 km = cluster.KMeans(n_clusters=k) 
 km.fit(Df_final) 
 sse.append(km.inertia_)



# from matplotlib import pyplot as plt 
# get_ipython().run_line_magic('matplotlib', 'inline')



plt.xlabel('K') 
plt.ylabel('Sum of squared error') 
plt.plot(k_rng,sse)





#DBSCAN
from sklearn import cluster

dbsc = cluster.DBSCAN(eps = .005, min_samples = 15).fit(Df_final)
labels=pd.DataFrame(dbsc.labels_)
labels_New=dbsc.labels_
labels.columns = ['Cluster']

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels_New)) - (1 if -1 in labels_New else 0)

n_noise_ = list(labels_New).count(-1)




# we merge this dataframe with df
prediction = pd.concat([Df_final, labels], axis = 1)

#print(prediction)

# We store the clusters
cluster_list=[]

for i in range(n_clusters_):
    cluster_list.append(prediction.loc[prediction.Cluster == i].values)
    

#print(dunn(cluster_list))




dframe= pd.concat([df_ex_preprocessing, labels], axis = 1)



from fastapi.encoders import jsonable_encoder
input_name="Pixel Draw - Number Art Coloring Book"
#Generate relevant app details and top randomly generated app names for the input app cluster
def generateAppCluster(input_name):

    #extract cluster
    cluster=dframe.loc[dframe['App']==input_name]['Cluster']

    app_details=dframe.loc[dframe['App']==input_name]
    app_details=pd.DataFrame(app_details[['App','Category','Rating','Last Updated','Content Rating']])
    app_details_list = app_details.values.tolist()

    #app_details = jsonable_encoder(tuple(app_details[1]))
    #app_details = app_details.to_json(orient = 'columns')
    print("App Details.. :" ,app_details)
    app_details=[]
    for i in app_details_list[0]:
        app_details.append(i)
    print(app_details)
    cluster=list(cluster)
    print(cluster)

    #Generate tpo 10 apps
    cluster_apps=pd.DataFrame(dframe.loc[dframe['Cluster']==cluster[0]])
    cluster_apps=pd.DataFrame(cluster_apps.loc[dframe['Rating']>4.5])
    cluster_apps=pd.DataFrame(cluster_apps[['App','Rating']])
    cluster_apps= cluster_apps.sample(n=10)
    cluster_apps_list = cluster_apps.values.tolist()
    cluster_apps=[]
    for i in cluster_apps_list:
          cluster_apps.append(i)
    print(cluster_apps)
    #cluster_apps=cluster_apps.to_json(orient = 'columns')
    
    return app_details,cluster_apps




app=generateAppCluster(input_name)
print(type(app))










# from joblib import dump





# dump(dbsc, 'server/models/dbscan.joblib') 







######################
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from model_loader import ModelLoader
# from train_parameters import TrainParameters
from AppName import AppName
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import json

origins = [
    "http://localhost:8000",
    "http://127.0.0.1:8000/predict",
    "http://127.0.0.1:8000/load",
    "http://localhost:4200"
]


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(data:AppName):
    print("Predicting")
    print(type(data))
    app,cluster_apps =generateAppCluster(data.name)
    json_compatible_app_data = jsonable_encoder(app)
    json_compatible_cluster_data = jsonable_encoder(cluster_apps)
    return_output=json_compatible_app_data,json_compatible_cluster_data
    json_compatible_item_data = jsonable_encoder(return_output)
    return JSONResponse(content=json_compatible_item_data)

   
