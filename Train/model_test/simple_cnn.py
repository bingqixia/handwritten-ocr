import argparse
import numpy

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from emnist_data_handler import read_emnist
from save_model import save

K.set_image_dim_ordering('th')


def build_net(training_data, model_name='model', epochs=10):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping = training_data

    # reshape to be [samples][pixels][width][height]
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # create model
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=200, verbose=1)
    print(model.summary())

    modelDir = 'models/'+model_name
    batch_size=200
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)
    esCallBack = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min')
    mcCallBack = keras.callbacks.ModelCheckpoint(filepath=modelDir+'/model.h5', monitor='val_loss', 
                                                verbose=0, save_best_only=True, 
                                                save_weights_only=False, 
                                                mode='auto', period=1)
    # Callback for analysis in TensorBoard
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tbCallBack])
    
    # Callback for early-stop
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[esCallBack])

    # Callback for ModelCheckpoint
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[mcCallBack])

    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    
    model_yaml = model.to_yaml()
    with open(modelDir+"/model.yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    
    # Final evaluation of the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print("simple_cnn Error: %.2f%%" % (100-scores[1]*100))
    return


if __name__ == '__main__':

    training_data = read_emnist('EMNIST')
    model = build_net(training_data, 'simple_cnn', 15)

