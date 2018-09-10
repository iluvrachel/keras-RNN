

# please note, all tutorial code are running under python3.5.
# If you use the version like python2.7, please modify the code accordingly

# 8 - RNN Classifier example

# to try tensorflow, un-comment following two lines
# import os
# os.environ['KERAS_BACKEND']='tensorflow'
import copy
import os
import numpy as np
# np.random.seed(1996)  # for reproducibility
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, Dropout,LSTM, GRU, Input
from keras.optimizers import Adam
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


TIME_STEPS = 990     # same as the height of the image
INPUT_SIZE = 99     # same as the width of the image
BATCH_SIZE = 1
BATCH_INDEX = 0
OUTPUT_SIZE = 4
CELL_SIZE = 128
LR = 0.00005
nb_epoch = 20

train_subject_ids = [1,6,7,8,9,11]
# actions = ["walking", "eating", "smoking", "discussion",  "directions",
#               "greeting", "phoning", "posing", "purchases", "sitting",
#               "sittingdown", "takingphoto", "waiting", "walkingdog",
#               "walkingtogether"]

actions = ["greeting", "posing","takingphoto","smoking"]

tf.app.flags.DEFINE_string("data_dir", os.path.normpath("./data/h3.6m/dataset"), "Data directory")
FLAGS = tf.app.flags.FLAGS


def readCSVasFloat(filename):

    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    returnArray = returnArray[:990]
    return returnArray


# def load_label():
#     label = [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14]
#
#     # label_onehot = tf.one_hot(label, 15, on_value=1, off_value=0, axis=1)  # TODO
#     # print(label_onehot.shape)
#     return label

def load_data(path_to_dataset, subjects, actions):

    nactions = len(actions)

    completeData = []

    for subj in subjects:
        for action_idx in np.arange(len(actions)):

            action = actions[action_idx]

            for subact in [1, 2]:  # subactions

                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))  # TODO

                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                action_sequence = readCSVasFloat(filename)

                n, d = action_sequence.shape

                print(n)
                print(d)

                if len(completeData) == 0:
                    completeData = copy.deepcopy(action_sequence)
                else:
                    completeData = np.append(completeData, action_sequence, axis=0)

    print(completeData.shape)
    return completeData

# def label_process(batch_size, data, label):
#     returnArray = []
#     for n in range(0, data.shape[0], batch_size):
#         i = 1
#         # x = data[n:n + batch_size, :, :]
#         y = label[:i]
#         y = np.tile(y, (33, 1))
#         i = i + 1
#         returnArray.append(y)
#     return returnArray

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

history = LossHistory()

# early_stopping =EarlyStopping(monitor='val_loss', patience=10)

# X shape (60,000 28x28), y shape (10,000, )
X_train = load_data(FLAGS.data_dir, train_subject_ids, actions)
y_train = np.loadtxt('train_label.txt')


X_test = load_data(FLAGS.data_dir, [5], actions)
y_test = np.loadtxt('test_label.txt')


# data pre-processing
X_train = X_train.reshape(-1, TIME_STEPS, INPUT_SIZE)      # normalize
print("train shape")
print(X_train.shape)
print(y_train.shape)
X_test = X_test.reshape(-1, TIME_STEPS, INPUT_SIZE)        # normalize
print("test shape")
print(X_test.shape)
print(y_test.shape)
y_train = np_utils.to_categorical(y_train, num_classes=OUTPUT_SIZE)

y_test = np_utils.to_categorical(y_test, num_classes=OUTPUT_SIZE)

# build RNN model
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    # for batch_input_shape, if using tensorflow as the backend, we have to put None for the batch_size.
    # Otherwise, model.evaluate() will get error.
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
))

#######################################################################

# LSTM cell
# model.add(LSTM(512, activation='tanh',input_shape=(TIME_STEPS, INPUT_SIZE),return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(256, activation='tanh'))
# model.add(Dropout(0.2))
# model.add(Dense(5,activation='softmax'))

#######################################################################


#GRU cell
# model.add(GRU(32, activation='tanh',input_shape=(TIME_STEPS, INPUT_SIZE),return_sequences=True))
# model.add(Dropout(0.4))
# model.add(GRU(16, activation='tanh'))

# output layer
model.add(Dense(OUTPUT_SIZE))  # fc layer
model.add(Activation('softmax'))

# optimizer
adam = Adam(LR)
# rmsprop = RMSprop(lr=LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training
# for step in range(100001):
#     # data shape = (batch_num, steps, inputs/outputs)
#     X_batch = X_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :, :]
#     Y_batch = y_train[BATCH_INDEX: BATCH_INDEX+BATCH_SIZE, :]
#     cost = model.train_on_batch(X_batch, Y_batch)
#     BATCH_INDEX += BATCH_SIZE
#     BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
#
#     if step % 500 == 0:
#         cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=False)
#         print('test cost: ', cost, 'test accuracy: ', accuracy)


model.fit(X_train, y_train,
            batch_size=BATCH_SIZE, nb_epoch=nb_epoch,
            verbose=1,
            shuffle=True,
            validation_data=(X_test, y_test),
            callbacks=[history])


score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

history.loss_plot('epoch')