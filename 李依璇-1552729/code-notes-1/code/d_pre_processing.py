# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 07:53:38 2018

@author: lenovo
"""


#anomaly sample_oversampling

#dataformimport pymysql
from pymysql import cursors

# Connect to the database
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='127Reborn',
                             db='anomaly_detection',
                             charset='utf8mb4',
                             cursorclass=cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        # Read a single record
        sql = "select * from `phase2_train`"
        cursor.execute(sql)
        result = cursor.fetchall()
        print('#result:',len(result))
finally:
    connection.close()

# =============================================================================
# import pandas as pd
# a['timestamp']=pd.to_datetime(int(a['timestamp']))
# print(a)
# =============================================================================
    
    
# =============================================================================
# data_import
# =============================================================================
    
import pandas as pd
anomaly=pd.read_csv(r"E:\phase2_train.csv")

# =============================================================================
# data_visualization
# =============================================================================    

import matplotlib.pyplot as plt
a=anomaly[2000:2100]
a_exp=a[a['label']==1]
a_nrm=a[a['label']==0]
plt.plot(a_nrm['timestamp'],a_nrm['value'], 'go--', linewidth=2, markersize=1)
plt.plot(a_exp['timestamp'],a_exp['value'], 'ro--', linewidth=2, markersize=1)
plt.show()

# =============================================================================
# preprocess
# 
# data_resampled ----》X_resampled,y_resampled (3655640,2);(3655640,)
# weight_adjust ----?
# =============================================================================


from imblearn.over_sampling import SMOTE
from collections import Counter
X=anomaly[['timestamp','value']]
y=anomaly['label']
ratio={1:int(len(y[y==0])*0.25)}
X_resampled, y_resampled = SMOTE(ratio=ratio,kind='borderline1').fit_sample(X, y)
print(Counter(y_resampled).items()) #dict_items([(0, 2924512), (1, 731128)])

X_df=pd.DataFrame(X_resampled)
y_df=pd.DataFrame(y_resampled)
anomaly_resampled=pd.concat([X_df,y_df],axis=1) 
anomaly_resampled.columns=['timestamp','value','label']

#

from sklearn import preprocessing

value_scaled = preprocessing.scale(anomaly_resampled['value'])
anomaly_resampled['value']=pd.Series(value_scaled)
# =============================================================================
# feature_extraction
# =============================================================================

value_groupby_mean=anomaly_resampled[['value','label']].groupby(by='label').mean()




    