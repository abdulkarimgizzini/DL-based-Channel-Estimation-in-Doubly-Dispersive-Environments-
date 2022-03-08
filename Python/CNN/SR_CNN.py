import os
import tensorflow as tf
import numpy
# from models import SRCNN_train, SRCNN_model, SRCNN_predict, DNCNN_train, DNCNN_model, DNCNN_predict
from scipy.io import loadmat
from keras.models import Sequential, Model
from keras.layers import Convolution2D, Input, BatchNormalization, Conv2D, Activation, Lambda, Subtract, \
    Conv2DTranspose, PReLU
from keras.regularizers import l2
from keras.layers import Reshape, Dense, Flatten
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from scipy.io import loadmat
import keras.backend as K
# from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
import numpy as np
import math
from scipy import interpolate
SNR_array = [0, 5, 10, 15, 20, 25, 30, 35, 40]

nSym = 100
N_Train = 8000
N_Test = 2000
snr = 9
epoch = 200
batch_size = 128
val_split = 0.25
Training_SNR = SNR_array[int(snr) - 1]
SNR_index = np.arange(1, 8)


def SRCNN_model(nSym):
    model = Sequential()
    model.add(
        Conv2D(32, kernel_size=(9, 9), activation='relu', kernel_initializer='he_normal', input_shape=(104, nSym, 1),
               padding='same'))
    model.add(Conv2D(16, kernel_size=(1, 1), activation='relu', kernel_initializer='he_normal', padding='same'))
    model.add(Conv2D(1, kernel_size=(5, 5), kernel_initializer='he_normal', padding='same'))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    return model


def SRCNN_train(X_train, Y_train, nSym, Training_SNR, batch_size, epoch, val_split):
    srcnn_model = SRCNN_model(nSym)
    print(srcnn_model.summary())
    model_path = './ALS_High_SRCNN_{}.h5'.format(Training_SNR)
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    srcnn_model.fit(X_train, Y_train, batch_size=batch_size, validation_split=val_split,
                    callbacks=callbacks_list, epochs=epoch, verbose=2)


def SRCNN_predict(nSym, C, scheme, Training_SNR, SNR_index, N_Test):
    for j in SNR_index:
        # Load Matlab DataSets
        mat = loadmat("/users/abdugizz/test/cnn_dataset/{}/{}_CNN_Dataset_{}.mat".format(C, scheme, snr))
        CNN_Dataset = mat['CNN_Dataset']
        CNN_Dataset = CNN_Dataset[0, 0]
        X_test = CNN_Dataset['Test_X']
        Y_test = CNN_Dataset['Test_Y']
        print('Loaded Testing Dataset: ', X.shape)
        print('Loaded Testing Dataset: ', Y.shape)

        CNN_X_test = X_test.transpose(2, 0, 1)
        CNN_X_test = CNN_X_test.reshape(N_Test, 104, nSym, 1)
        CNN_Y_test = Y_test.transpose(2, 0, 1)
        CNN_Y_test = CNN_Y_test.reshape(N_Test, 104, nSym, 1)

        model_path = '/users/abdugizz/test/cnn_trained_models/{}_{}_SRCNN_{}.h5'.format(C, scheme, Training_SNR)

        srcnn_model = load_model(model_path)
        Prediction_Y = srcnn_model.predict(CNN_X_test)

        result_path = '/users/abdugizz/test/results/{}_{}_SRCNN_Results_{}.pickle'.format(C, scheme, j)
        with open(result_path, 'wb') as f:
            pickle.dump([X_test, Y_test, Prediction_Y], f)

    for j in SNR_index:
        source_name = '/users/abdugizz/test/results/{}_{}_SRCNN_Results_{}.pickle'.format(C, scheme, j)
        dest_name = '/users/abdugizz/test/results_matlab/{}_{}_SRCNN_Results_{}.mat'.format(C, scheme, j)
        a = pickle.load(open(source_name, "rb"))
        scipy.io.savemat(dest_name, {
            '{}_{}_SRCNN_test_x_{}'.format(C, scheme, j): a[0],
            '{}_{}_SRCNN_test_y_{}'.format(C, scheme, j): a[1],
            '{}_{}_SRCNN_corrected_y_{}'.format(C, scheme, j): a[2]
        })
        print("Data successfully converted to .mat file ")


# Load Matlab DataSets
#mat = loadmat("./ALS_High_CNN_Dataset_9.mat")
#CNN_Dataset = mat['CNN_Dataset']
#CNN_Dataset = CNN_Dataset[0, 0]
#X = CNN_Dataset['Train_X']
#Y = CNN_Dataset['Train_Y']
#print('Loaded Training Dataset: ', X.shape)
#print('Loaded Training Dataset: ', Y.shape)

#CNN_X = X.transpose(2, 0, 1)
#CNN_X = CNN_X.reshape(N_Train, 104, nSym, 1)
#CNN_Y = Y.transpose(2, 0, 1)
#CNN_Y = CNN_Y.reshape(N_Train, 104, nSym, 1)



CNN_X = np.random.randint(10, size=(100, 104, 2, 1))
CNN_Y = np.random.randint(10, size=(100, 104, 2, 1))
print('SR-CNN Inputs: ', CNN_X.shape)
print('SR-CNN Outputs: ', CNN_Y.shape)




# SR_CNN Training
SRCNN_train(CNN_X, CNN_Y, nSym, Training_SNR, batch_size, epoch, val_split)
# SR_CNN Testing
#SRCNN_predict(nSym, C, scheme, Training_SNR, SNR_index, N_Test)