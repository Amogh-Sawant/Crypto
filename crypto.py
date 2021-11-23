from bs4 import BeautifulSoup
import requests
import lxml
import pandas as pd
from datetime import date
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import numpy as np
import random
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, BatchNormalization

today = (date.today()).strftime("%d%m%Y")

bitcoin_data = requests.get('https://coinmarketcap.com/currencies/bitcoin/historical-data/').text
soup = BeautifulSoup(bitcoin_data, 'lxml')

close_price = soup.find('div', class_ = 'priceValue___11gHJ').text
btc_data = soup.find_all('div', class_ = 'statsValue___2iaoZ')  
btc_low = soup.find('div', class_ = 'sc-16r8icm-0 gMZGhD nowrap___2C79N').text
btc_high = soup.find('div', class_ = 'sc-16r8icm-0 HwsGY nowrap___2C79N').text
market_cap = btc_data[0].text
volume = btc_data[2].text

dataset = f"cryptoData/BTC_USD.csv"
csv_data = pd.read_csv(dataset)
open_price = csv_data.iloc[-1,4]

last5p = int(len(csv_data)*0.05)

main_df = pd.DataFrame(csv_data[0:len(csv_data)-last5p])
validation_df = pd.DataFrame(csv_data[len(csv_data)-last5p:])

SEQ_LEN = 60

# print(f"{today}, {open_price}, {btc_high}, {btc_low}, {close_price}, {market_cap}, {volume}") 

def preprocess_df(df):
    df = df.drop('Date', 1)
    sequencial_data = []
    x = []
    y = [] 

    scaler = MinMaxScaler(feature_range=(0,1))
    # print(df.max())
    # print(df.min())
    for col in df.columns:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
        # df[col] = df[col].pct_change()
        # df[col]= preprocessing.scale(df[col].values)
        

    prev_days = deque(maxlen=SEQ_LEN) 

    for row in df.values:
        prev_days.append([data for data in row[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequencial_data.append([np.array(prev_days), row[-1]])
    
    random.shuffle(sequencial_data)
    for seq, target in sequencial_data:
        x.append(seq)
        y.append(target)

    return x, y

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_df)

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.Dropout(0.2)
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.Dropout(0.2)
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.Dropout(0.2)
model.add(BatchNormalization())

6:28