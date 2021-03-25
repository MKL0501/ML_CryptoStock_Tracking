

# In[Imports]

import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# In[Parameters]

#define the ticker symbol
ticker_symbols = ['TSLA','GOOG','AMZN','AAPL']

#number of days
window_size = 365

#split ratio points of dataset
train_stop = 0.7
test_start = 0.9

#number of training cycles/iterations
epoch = 50

#early staopping trigger
wait = 10

#training cycles before weights are updated
batch = 32

#how many epochs between checking validation data
val_freq = 1

# In[Create DataFrame]

df_dict = {}
for symbol in ticker_symbols:
    #get data on each ticker
    ticker_data = yf.Ticker(symbol)
    
    #get the historical prices for each ticker
    df_dict[symbol] = ticker_data.history(period='1d', start='2010-1-1', end='2021-1-1')

#make dataframe
data = pd.concat(df_dict, axis = 1)

#replace nans with zeros
data = data.fillna(0)

# In[Normalize Data]

n = len(data)

train_data_mean = data[0 : int(n * train_stop)].mean()
train_data_std = data[0 : int(n * train_stop)].std()

#make a normal distribution
scaled_data = (data - train_data_mean) / train_data_std

#have to remove nan again, since normalizing causes more
scaled_data = scaled_data.fillna(0)

# In[Create Targets]

scaled_target = {}
for i in range(scaled_data.shape[0] - 1):
    closes = []
    for ticker in ticker_symbols:
        closes.append(scaled_data[(ticker, 'Close')][i + 1])
    scaled_target[scaled_data.index[i]] = closes

#create dataframe of targets
scaled_target = pd.DataFrame.from_dict(scaled_target ,orient = 'index', columns = ticker_symbols)

#drop last day, since we don't know the closing for tomorrow
scaled_data = scaled_data[: -1]

# In[Create Windows]

x = []
y = []

#goes from end of first window to end of data
for i in range(window_size, scaled_data.shape[0]):
    x.append(scaled_data[i - window_size : i])
    y.append(scaled_target[i : i + 1])
    
x = np.array(x)
y = np.array(y)


# In[Split Data]

#same names of labels
data_column_indices = {name : i for i, name in enumerate(scaled_data.columns)}
target_column_indices = {name : i for i, name in enumerate(scaled_target.columns)}

x_train = x[0 : int(n * train_stop)]
x_val = x[int(n * train_stop) : int(n * test_start)]
x_test = x[int(n * test_start) :]

y_train = y[0 : int(n * train_stop)]
y_val = y[int(n * train_stop) : int(n * test_start)]
y_test = y[int(n * test_start) :]

num_features = scaled_data.shape[1]

# In[Create Model]

model = Sequential([
                    LSTM(units = 512, activation = 'tanh', return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])),
                    #Dropout(0.2),
                          
                    LSTM(units = 512, activation = 'tanh', return_sequences = True),
                    ##Dropout(0.2),
                    
                    LSTM(units = 512, activation = 'tanh', return_sequences = True),
                    #Dropout(0.2),
                          
                    LSTM(units = 512, activation = 'tanh'),
                    #Dropout(0.2),
                          
                    Dense(units = len(ticker_symbols), activation='linear')
                   ])

model.compile(
              #loss = 'mean_squared_error',
              loss = 'mae',
              optimizer = 'adam',
              metrics=['accuracy']
             )

model.summary()

# In[Train Model]

history = model.fit(
                    x = x_train,
                    y = y_train,
                    epochs = epoch,
                    batch_size = batch,
                    shuffle = False,
                    validation_data = (x_val, y_val),
                    validation_freq = val_freq,
                    callbacks = [EarlyStopping(patience = wait)]
         )

# In[Test Model]

model.evaluate(
               x = x_test,
               y = y_test
              )

y_pred = model.predict(x_test)

# In[Graph Results]

#plot loss
plt.plot(
         history.history['loss'],
         label = 'Training Loss',
         color = 'green',
         linestyle = 'dashed'
        )

plt.plot(
         history.history['val_loss'],
         label = 'Validation Loss',
         color = 'blue',
         linestyle = 'dashed'
        )

plt.legend()
plt.show()

#plot accuracy
plt.plot(

         history.history['accuracy'],
         label = 'Training Accuracy',
         color = 'green',
         linestyle = 'solid'
        )

plt.plot(
         history.history['val_accuracy'],
         label = 'Validation Accuracy',
         color = 'blue',
         linestyle = 'solid'
        )

plt.legend()
plt.show()


#plot stock predictions
y_test_2d = y_test.reshape(y_test.shape[0], y_test.shape[2])
y_pred = np.array(y_pred)

for i in range(len(ticker_symbols)):
    plt.plot(y_test_2d[:, i], color = 'green', label = 'Actual ' + ticker_symbols[i] + ' Price')
    plt.plot(y_pred[:, i], color = 'blue', label = 'Predicted ' + ticker_symbols[i] + ' Price')
    plt.title(ticker_symbols[i] + ' Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()