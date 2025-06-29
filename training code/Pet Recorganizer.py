
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

#processing  the images like VGG16
datagen= ImageDataGenerator(
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
)
#making a callback function

early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)
#these are the classes which i have
Classes=[
    'Cat',
    'Cow',
    'Dog',
    'Fish',
    'Goat'
]
#path of the images
train_path='project/dataset/train'
valid_path='project/dataset/valid'
train_batch=datagen.flow_from_directory(directory=train_path,target_size=(224,224),classes=Classes,batch_size=32)
valid_batch=datagen.flow_from_directory(directory=valid_path,target_size=(224,224),classes=Classes,batch_size=32 )

#Creating a sequential model
model= Sequential(
    [
        Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same',input_shape=(224,224,3))  ,
        MaxPool2D(pool_size=(2,2),strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=2),
        Flatten(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(5 , activation='softmax')

    ]
)


model.compile (optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_batch,validation_data=valid_batch,epochs=22  ,    callbacks=[early_stop],verbose=1 )
model.save('Final.h5')
model.save('Final.keras')