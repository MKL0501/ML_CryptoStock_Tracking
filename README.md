# ML For Stock Market Prediction Using RNN/LSTM
The project is inteded to predict stock prices based on the market data available. 

# Description
The project uses tensorflow and keras for machine learning, and yahoo finance API for pulling the data.
By default, it will grab all the data of TSLA stock to train, but you can change the ticker name within the code.
It is somewhat accurate, though I would not give it money.

# Requirements
* Python 3
* Tensorflow 
* Keras
* matplotlib
* numpy
* sklearn
* pandas
* yahoo_fin

# Execution
Use "Python3 trader1.py" to run. On a separate terminal, you can use "tensorboard --logfile "logs" on the same folder in order to use tensorboard to check the results of the training. The trained model and the data is also saved on individual folders. More details can be found in the comments of the code.
