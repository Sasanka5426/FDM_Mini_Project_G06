# # Commented out IPython magic to ensure Python compatibility.
# import numpy as np
# import pandas as pd
# #import sklearn as sl
# from scipy.optimize import minimize
# import matplotlib.pyplot as plt
# import os
# # %matplotlib inline

# class Preprocessing():
#     #import data
#     df_ex_preprocessing = pd.read_csv('data/googleplaystore.csv')




#     #Drop unwanted columns
#     df = df_ex_preprocessing.drop(["App","Current Ver"],1)


#     #delete Type,Content Rating
#     df_train = df.copy()
#     for i in ['Type','Content Rating','Android Ver']:
#         df_train = df_train.drop(df_train.loc[df_train[i].isnull()].index,0)
#     df_train.info()



#     #fill rating null values with mean quartiles
#     x = sum(df_train.Rating.describe()[4:8])/4
#     df_train.Rating = df_train.Rating.fillna(x)
#     #print("Dataset contains ",df_train.isna().any().sum()," Nan values.")


#     from sklearn.preprocessing import OneHotEncoder



#     #Hot encode the 'Category' and 'Genre' columns
#     encode = OneHotEncoder()

#     data_encoded = encode.fit(df_train[['Category','Genres']]).transform(df_train[['Category','Genres']])
#     temp = pd.DataFrame(data_encoded.toarray(),columns=encode.get_feature_names())

#     df_train = pd.concat([temp, df_train[['Rating', 'Reviews', 'Size', 'Installs', 'Type',
#                                         'Price', 'Content Rating', 'Last Updated', 'Android Ver']]], axis = 1, join = 'inner')





#     #Map the content rating column
#     df_train['Content Rating'] = df_train['Content Rating'].map({'Unrated':0.0,
#                                                     'Everyone':1.0,
#                                                     'Everyone 10+':2.0,
#                                                     'Teen':3.0,
#                                                     'Adults only 18+':4.0,
#                                                     'Mature 17+':5.0})
#     df_train['Content Rating'] = df_train['Content Rating'].astype(float)



#     #change type to float for Review
#     df_train['Reviews'] = df_train['Reviews'].astype(float)


#     #clean 'M','k', fill 'Varies with device' with median and transform to float 
#     lists = []
#     for i in df_train["Size"]:
#         if 'M' in i:
#             i = float(i.replace('M',''))
#             i = i*1000000
#             lists.append(i)
#         elif 'k' in i:
#             i = float(i.replace('k',''))
#             i = i*1000
#             lists.append(i)
#         else:
#             lists.append("Unknown")

   
#     # k = pd.Series(lists)
    
#     median_ = k[k!="Unknown"].median()
#     k = [median_ if i=="Unknown" else i for i in k]
#     for i in k:
#         if i=="Unknown" :
#             i=median_
#         else :
#             i
        
        
#     # df_train["Size"] = k

#     # del k,median_,lists





#     #clean '$' in Price and transform to float 
#     df_train['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in df_train['Price'] ]



#     #Remove '+' and ',' from installs
#     df_train["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in df_train["Installs"] ]


#     #Map the 'Type' column 
#     df_train.Type = df_train.Type.map({'Free':0,"Paid":1})


#     from sklearn.preprocessing import LabelEncoder

#     le=LabelEncoder()

#     #transform ' Android Ver ' column
#     df_train['Android Ver']=le.fit(df_train['Android Ver']).transform(df_train['Android Ver'])





#     from datetime import datetime


#     df_train["Last Updated"] = [datetime.strptime(i, '%B %d, %Y') for i in df_train["Last Updated"]]


#     df_train["Last Updated"] = [i.year for i in df_train["Last Updated"]]




#     #transform ' Last Updated ' column
#     df_train['Last Updated']=le.fit(df_train['Last Updated']).transform(df_train['Last Updated'])


#     df_train['Last Updated'].unique()



#     df_train.isna().any().sum()



#     #Normalizing the column values
#     from sklearn import preprocessing

#     normalized_X = preprocessing.normalize(df_train)
#     Df_final=pd.DataFrame(normalized_X)
#     print(Df_final)

