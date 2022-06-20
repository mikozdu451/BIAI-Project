import tensorflow as tf
from sklearn.metrics import f1_score
from keras import optimizers
from keras.models import Sequential
from keras.models import Sequential, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



class stop_training_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        data = str(logs.get('loss')) + ' ' + str(logs.get('custom_f1score')) + ' ' + str(logs.get('val_loss')) + ' ' + str(logs.get('val_custom_f1score')) + '\n'
        f.write(data)
        print(data)

        #if(logs.get('val_custom_f1score') > 0.99):
        #      self.model.stop_training = True


def f1score(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro')


def custom_f1score(y, y_pred):
    return tf.py_function(f1score, (y, y_pred), tf.double)

def store_keras_model(model, model_name):
    model_json = model.to_json() # serialize model to JSON
    with open("./{}.json".format(model_name), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./{}.h5".format(model_name)) # serialize weights to HDF5
    print("Saved model to disk")


def load_keras_model(model_name):
    # Load json and create model
    json_file = open('./{}.json'.format(model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights("./{}.h5".format(model_name))
    return model


train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
path = 'C:/Users/pumpk/Desktop/git/BIAI/ALPR/data'
train_generator = train_datagen.flow_from_directory(
        path+'/train',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28
        batch_size=1,
        class_mode='sparse')

validation_generator = train_datagen.flow_from_directory(
        path+'/val',  # this is the target directory
        target_size=(28,28),  # all images will be resized to 28x28 batch_size=1,
        class_mode='sparse')

K.clear_session()
model = Sequential()
#model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
#model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
#model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.4))
#model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', metrics=[custom_f1score])



#model.summary()
f = open('dataForGraphModel3.txt', 'a')
i = 1
batch_size = 1
callbacks = [stop_training_callback()]
model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples // batch_size,
      validation_data=validation_generator,
      epochs=80, verbose=1, callbacks=callbacks)

store_keras_model(model, 'model_LicensePlate3')


f.close()
