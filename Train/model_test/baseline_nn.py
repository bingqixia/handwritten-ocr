from keras.layers import Dense
from keras.models import Sequential
from keras.utils import np_utils
import keras
from emnist_data_handler import read_emnist
from save_model import save


def build_net(training_data, model_name='model', epochs=10):

    # Initialize data
    (x_train, y_train), (x_test, y_test), mapping = training_data

    # flatten 28*28 images to a 784 vector for each image
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels)
    x_test = x_test.reshape(x_test.shape[0], num_pixels)

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    # create model
    model = Sequential()
    model.add(Dense(784, input_dim=784, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    batch_size=200
    modelDir = 'models/'+model_name
    # Fit the model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
    
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
    #save_model(model, modelDir+'/model.h5')
    #save(model, mapping, model_name)
    return


if __name__ == '__main__':
    training_data = read_emnist('EMNIST')
    model = build_net(training_data, 'baseline_nn', 12)
