# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 18:26:49 2018

@author: lenovo
"""

import pandas as pd
df1=pd.DataFrame([[1,2],[2,3],[3,4],[4,5]],columns=['0','1'])
df2=pd.DataFrame([1,2],columns=['2'])
pd3=pd.DataFrame([])
df3=pd.concat([df1,pd.DataFrame(df2.ioc[0,0] if df1['1']<3 else df2.iloc[1,0]) ],axis=1)