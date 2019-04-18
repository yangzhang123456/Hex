# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''
A股数据 tushare
'''

'''
https://www.cnblogs.com/fuqia/p/9064750.html
cross_validation 0.18版本已被sklearn废弃
'''
import os
import pandas as pd
import tushare as ts
import math
import numpy as np
from sklearn import preprocessing,model_selection,svm#向量机
from sklearn.linear_model import LinearRegression


#print(open('D:/zy.txt','r').read())

df = ts.get_hist_data('000001')#从tushare获取历史开盘数据数据，招行


df=df[['open','high','close','low','volume']]#截取需要的字段

df['High_Low_Pct']=(df['high']-df['low'])/df['low']*100#计算振幅
df['Change_Pct']=(df['close']-df['open'])/df['open']*100

df=df[['close','High_Low_Pct','Change_Pct','volume']]

pd.set_option('display.max_rows',1000)#设置尺寸
pd.set_option('display.max_columns',1000)

#file = open('D:/zy.txt','w')
#file.write(str(df.head()))
#file.close()

#print(df.head())

future_value='close'
df.fillna(value=-99999,inplace=True)#异常值替换空值
how_far_I_want_to_forcast=math.ceil(0.01*len(df))#1%预测值，近期预测,float取整
#print(how_far_I_want_to_forcast)
df['label']=df[future_value].shift(-how_far_I_want_to_forcast)#尾部数据
#print(df.head())#七天预测值

df.dropna(inplace=True)
#drop none valuable shift and drop process data
#print(df)
#print(df.head())

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)#预处理
X_Recent_Real_Data = X[-how_far_I_want_to_forcast:]
Y = np.array(df['label'])

X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=0.5)#25%用作训练机，75&用作检测机洗牌，希望得到对未来预测有用的数据，而不是只符合过去规律

#black_box = LinearRegression()  #线性回归
black_box = LinearRegression(n_jobs=-1)
black_box.fit(X_train,Y_train)
#accuracy = black_box.score(X_test,Y_test)
#print(accuracy)


forecast_set = black_box.predict(X_Recent_Real_Data)
print(forecast_set)