import pandas as pd
import numpy as np
data = pd.read_csv('kc_house_data.csv')
p1 = data['yr_built'].to_numpy()
p2 = data['yr_renovated'].to_numpy()
for i in range(0,21613):
 if(p2[i]>p1[i]):
    p1[i]=p2[i]

data['yr_effective'] = p1
data['sqft_total'] = data['sqft_living'] + data['sqft_lot'] + data['sqft_above'] + data['sqft_basement'] + data['sqft_living15'] + data['sqft_lot15']
dataSub = data[['price','bedrooms','bathrooms','floors','waterfront','view','sqft_total','yr_effective','lat','long']]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x = dataSub.iloc[:,1:10]
y = dataSub['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=30)
reg = LinearRegression()
reg.fit(x_train,y_train)
predict = reg.predict(x_test)
import pickle
pickle.dump(reg,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))