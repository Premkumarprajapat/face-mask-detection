import os 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D

img_size = 224
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split=0.2,
    horizontal_flip = True
)

train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(img_size,img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

base_model = MobileNetV2(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
base_model.trainable = False

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(128,activation='relu')(x)
output = Dense(2,activation='softmax')(x)

model = Model(imputs=base_model.input,outputs=output)
model.compile(optimizer='adam',loss='categorial_crossentropy',metrics=['accuracy'])

model.fit(train_generator,validation_data=val_generator,epochs=10)

model.save("mask_detector_model.h5")
