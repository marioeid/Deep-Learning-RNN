                        # Recurent neural network

                        # pre processing 
# import libraries 

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# import the training set 

# stock prices from 2012 to 2016 
dataset_train=pd.read_csv('train.csv')
training_set=dataset_train.iloc[:,1:2].values

# feature scalling

# it's recommended to use normalization for RNN if the activation function is sigmoid 
from sklearn.preprocessing import MinMaxScaler 
sc=MinMaxScaler(feature_range=(0,1))

# it's recommended to keep your training set variable
# sc.fit means that it will get the max and min to apply the normlization formula 
training_set_scaled=sc.fit_transform(training_set)

# creating a data structure with 60 time steps and 1 output 
X_train=[]
y_train=[]
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train,y_train=np.array(X_train),np.array(y_train)

# reshaping use it to add a new dim 
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

                        #   building  our RNN 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM

regressor= Sequential()
# adding the first LSTM layer and some drop out regularization 

# the first argument is the number of units which is the number of memory units you want to have in this layer
# the second argument is the retrun sequcences we will set it to true which will return a value to the next one so we will add to every one expect the last one
# the thirs argument is the input shape of only the time steps and the indicators for prediction

regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

# adding the second LSTM layer and some drop out regularization 
# we don't need to define the input shape after adding the first layer
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

# adding the second LSTM layer and some drop out regularization 

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))


# adding the second LSTM layer and some drop out regularization 
 # the defult return sequences is false we need to make it fals cause there's no layers after the last layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# adding the output layer we are predicting the stock price 1 value
regressor.add(Dense(units=1))

# compiling our regressor 
# loss is mse not binary entropy cause it's regression not classification 
regressor.compile(optimizer='adam',loss='mean_squared_error')


# fitting our RNN
# updating the wait every 32 obseravtions 
regressor.fit(X_train,y_train,epochs=100,batch_size=32)
 


# getting the real google stock price of 2017
# contains stock prices for jan 2017
dataset_test=pd.read_csv('test.csv')
real_stock_price=dataset_test.iloc[:,1:2].values

# getting the predicted stock price of 2017
# we need to get the 60 previous input for each day for january 2017 (input for each prediction)
# some of them will be in the training set and some in the test set 
# we need the test set and training we should never change the test set value so we will use the data frames here

#vecrtical concatination
dataset_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)
inputs=dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

X_test=[]
for i in range(60,80):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)
# reshaping use it to add a new dim 
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predected_stock_price=regressor.predict(X_test)

# visulising the results 
