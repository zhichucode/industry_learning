# import packages
# !pip install pandas
# !pip install xlrd==1.2.0
# !pip install openpyxl
# !pip install matplotlib
# !pip install keras
# !pip install tensorflow
# !pip install tqdm
# !pip install numba==0.53

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import librosa
import librosa.display

import logging
import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger.info("logger system works!")

def load_excel(filename):
    data = pd.read_excel(filename, index_col=None, usecols=[1, 2, 3])
    data = data.dropna(how='any').iloc[::-1]
    data = data.loc[~(data==0).all(axis=1)]
    xlist = reject_outliers(data['1号球磨轴承振动_X'].tolist())
    ylist = reject_outliers(data['1号球磨轴承振动_Y'].tolist())
    zlist = reject_outliers(data['1号球磨轴承振动_Z'].tolist())
    return xlist, ylist, zlist

def reject_outliers(data, m=2):
    mean = np.mean(data)
    std = np.std(data)
    m = 4
    logger.info("max: {}".format(mean+m*std))
    for i in range(len(data)) :
        if np.abs(data[i]) > mean + m*std or data[i] < 0 :
            logger.info("reject outlier data: {}".format(data[i]))
            data[i] = data[i-1]
    return data

import pickle
def save_pickle(filename, save_data):
    logger.info("save pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)

def load_pickle(filename):
    logger.info(f"load pickle <- {filename}")
    with open(filename, 'rb') as lf:
        load_data = pickle.load(lf)
    return load_data

import glob, os

x_pickle = "pickle/x.pickle"
y_pickle = "pickle/y.pickle"
z_pickle = "pickle/z.pickle"

# pickle
if os.path.exists(x_pickle) and os.path.exists(y_pickle) and os.path.exists(z_pickle):
    xlist_sum = load_pickle(x_pickle)
    ylist_sum = load_pickle(y_pickle)
    zlist_sum = load_pickle(z_pickle)
else:
    xlist_sum = ylist_sum = zlist_sum = []
    for file in tqdm(glob.glob("data/excels/*.xlsx")):
        logger.info("Load data from file {}".format(file))
        xlist, ylist, zlist = load_excel(file)
        xlist_sum.extend(xlist)
        ylist_sum.extend(ylist)
        zlist_sum.extend(zlist)
    save_pickle(x_pickle, xlist_sum)
    save_pickle(y_pickle, ylist_sum)
    save_pickle(z_pickle, zlist_sum)

def split_list(input_list, sublist_length):
    return [input_list[x:x+sublist_length] for x in range(0, len(input_list), sublist_length)]


sublist_size = 10000
xlists = split_list(xlist_sum, sublist_size)
ylists = split_list(ylist_sum, sublist_size)
zlists = split_list(zlist_sum, sublist_size)

fig, axs = plt.subplots(3)
fig.suptitle('the whole data of x, y, z')
axs[0].plot(xlist_sum)
axs[1].plot(ylist_sum)
axs[2].plot(zlist_sum)

count = 0
for i in ylist_sum:
    if i == 0:
        count += 1
logger.info(f"count of zeros: {count}")

from sklearn import metrics
from keras.models import Model
from keras.layers import Input, Dense

def keras_model(inputDim):
    inputLayer = Input(shape=(inputDim,))
    x = Dense(128, activation='relu')(inputLayer)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(  8, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(inputDim, activation=None)(x) # or sigmoid
    return Model(inputs=inputLayer, outputs=x)

model = keras_model(len(xlist_sum))
# model.summary()

class visualizer(object):
    def __init__(self):
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Test"], loc="upper right")

    def save_figure(self, filename):
        self.plt.savefig(filename)

from tqdm.keras import TqdmCallback
model_file = "model/model.hdf5"
visualizer = visualizer()

train_data = xlist_sum

if os.path.exists(model_file):
    model.load_weights(model_file)

else:
    model.compile(optimizer="adam", loss= "mean_squared_error")
    history = model.fit(train_data,
                        train_data,
                        epochs=100,
                        batch_size=10,
                        shuffle=True,
                        #validation_split=0.1,
                        verbose=0,
                        callbacks=[TqdmCallback(verbose=2)])
    visualizer.loss_plot(
        history.history["loss"], history.history["val_loss"]
    )
    visualizer.save_figure("model/history.png")
    model.save_weights(model_file)

