import sys
import numpy as np
from scipy.io import loadmat
from keras import Input, Model
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Conv2D, Activation, BatchNormalization, Subtract
from keras.optimizers import Adam
import pickle
import scipy.io
SNR_array = [0, 5, 10, 15, 20, 25, 30]

nSym = 100
N_Train = 8000
N_Test = 2000
C = sys.argv[1]
scheme = sys.argv[2]
snr = sys.argv[3]
epoch = int(sys.argv[4])
batch_size = int(sys.argv[5])
val_split = 0.25
Training_SNR = SNR_array[int(snr) - 1]
SNR_index = np.arange(1, 8)


def DNCNN_model(nSym):
    input = Input(shape=(104, nSym, 1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input)
    x = Activation('relu')(x)
    # 18 layers, Conv+BN+relu
    for i in range(18):
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3, 3), padding='same')(x)
    x = Subtract()([input, x])  # input - noise
    model = Model(inputs=input, outputs=x)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    print(model.summary())
    return model


def DNCNN_train(X_train, Y_train, nSym, Ch, mod, scheme, SNR, batch_size, epoch, val_split):
    dncnn_model = DNCNN_model(nSym)

    model_path = './test/cnn_trained_models/{}_{}_DNCNN_{}.h5'.format(C, scheme, Training_SNR)

    checkpoint = ModelCheckpoint(model_path, monitor='val_loss',
                                 verbose=1, save_best_only=True,
                                 mode='min')
    callbacks_list = [checkpoint]

    dncnn_model.fit(X_train, Y_train, batch_size=batch_size, validation_split=val_split,
                    callbacks=callbacks_list, epochs=epoch, verbose=2)


def DNCNN_predict(nSym, C, scheme, Training_SNR, SNR_index, N_Test):
    for j in SNR_index:
        # Load Matlab DataSets
        mat = loadmat("./test/cnn_dataset/{}/{}_CNN_Dataset_{}.mat".format(C, scheme, snr))
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

        model_path = './test/cnn_trained_models/{}_{}_DNCNN_{}.h5'.format(C, scheme, Training_SNR)

        srcnn_model = load_model(model_path)
        Prediction_Y = srcnn_model.predict(CNN_X_test)

        result_path = './test/results/{}_{}_DNCNN_Results_{}.pickle'.format(C, scheme, j)
        with open(result_path, 'wb') as f:
            pickle.dump([X_test, Y_test, Prediction_Y], f)

    for j in SNR_index:
        source_name = './test/results/{}_{}_DNCNN_Results_{}.pickle'.format(C, scheme, j)
        dest_name = './test/results_matlab/{}_{}_DNCNN_Results_{}.mat'.format(C, scheme, j)
        a = pickle.load(open(source_name, "rb"))
        scipy.io.savemat(dest_name, {
            '{}_{}_DNCNN_test_x_{}'.format(C, scheme, j): a[0],
            '{}_{}_DNCNN_test_y_{}'.format(C, scheme, j): a[1],
            '{}_{}_DNCNN_corrected_y_{}'.format(C, scheme, j): a[2]
        })
        print("Data successfully converted to .mat file ")


# Load Matlab DataSets
mat = loadmat("./test/cnn_dataset/{}/{}_CNN_Dataset_{}.mat".format(C, scheme, snr))
CNN_Dataset = mat['CNN_Dataset']
CNN_Dataset = CNN_Dataset[0, 0]
X = CNN_Dataset['Train_X']
Y = CNN_Dataset['Train_Y']
print('Loaded Training Dataset: ', X.shape)
print('Loaded Training Dataset: ', Y.shape)

CNN_X = X.transpose(2, 0, 1)
CNN_X = CNN_X.reshape(N_Train, 104, nSym, 1)
CNN_Y = Y.transpose(2, 0, 1)
CNN_Y = CNN_Y.reshape(N_Train, 104, nSym, 1)

print('DN-CNN Inputs: ', CNN_X.shape)
print('DN-CNN Outputs: ', CNN_Y.shape)

# DN_CNN Training
DNCNN_train(CNN_X, CNN_Y, nSym, C, scheme, Training_SNR, batch_size, epoch, val_split)
# DN_CNN Testing
DNCNN_predict(nSym, C, scheme, Training_SNR, SNR_index, N_Test)